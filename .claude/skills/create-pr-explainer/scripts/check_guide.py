# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright>=1.50,<2"]
# ///
"""Open a starter-based reviewer guide offline and optionally save figure previews."""

# Standard
from pathlib import Path
import argparse

# Third Party
from playwright.sync_api import Locator, Page, Request, sync_playwright


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _matches_language(item: Locator, language: str) -> bool:
    owner = item.evaluate("el => el.closest('main [lang]')?.getAttribute('lang')")
    return owner not in ("en", "zh-CN") or owner == language


def _check_layout(page: Page) -> None:
    problem = page.evaluate("""() => {
        const root = document.documentElement;
        if (root.scrollWidth > root.clientWidth + 1) return 'horizontal overflow';
        for (const a of document.querySelectorAll('a[href]')) {
            if (a.getAttribute('href').startsWith('#')) {
                const id = decodeURIComponent(a.hash.slice(1));
                if (!document.getElementById(id)) return 'broken link ' + a.hash;
            } else if (a.protocol === 'file:') {
                return 'local file link ' + a.getAttribute('href');
            }
        }
        for (const label of document.querySelectorAll('figure svg text')) {
            const r = label.getBoundingClientRect();
            const frame = label.ownerSVGElement.getBoundingClientRect();
            if (r.width && (r.left < frame.left - 1 || r.right > frame.right + 1 ||
                r.top < frame.top - 1 || r.bottom > frame.bottom + 1)) {
                return 'clipped label ' + label.textContent;
            }
        }
        return '';
    }""")
    _check(not problem, problem)


def _check_figures(
    page: Page, language: str, scripting: bool, preview: Path | None
) -> None:
    figures = page.locator("main figure")
    for index, figure in enumerate(figures.all(), 1):
        if not _matches_language(figure, language):
            continue
        _check(figure.is_visible(), f"figure {index} is hidden")
        panels = figure.locator("[data-view]").all()
        if scripting and figure.get_attribute("data-switch") is not None:
            controls = figure.locator("[data-controls] input[type=radio]").all()
            _check(len(controls) >= 2, f"figure {index} needs at least two choices")
            initial = [control for control in controls if control.is_checked()]
            _check(len(initial) == 1, f"figure {index} needs a selected choice")
            for choice, control in enumerate(controls, 1):
                control.press("Space")
                state = control.input_value()
                _check(
                    figure.get_attribute("data-state") == state,
                    f"inactive choice {state}",
                )
                for panel in panels:
                    visible = (
                        _matches_language(panel, language)
                        and panel.get_attribute("data-view") == state
                    )
                    _check(panel.is_visible() == visible, f"wrong panel for {state}")
                if panels:
                    _check(any(p.is_visible() for p in panels), f"no panel for {state}")
                _check_layout(page)
                if preview:
                    page.evaluate("document.activeElement.blur()")
                    figure.screenshot(path=preview / f"{index}-{choice}.png")
            initial[0].press("Space")
        elif not scripting:
            _check(
                not figure.locator("[data-controls]:visible").count(),
                "script-only controls are visible",
            )
            for panel in panels:
                if _matches_language(panel, language):
                    _check(panel.is_visible(), "static outcome is hidden")
        if preview and not panels:
            figure.screenshot(path=preview / f"{index}.png")


def _watch_page(page: Page, uri: str) -> list[str]:
    errors: list[str] = []

    def record_request(request: Request) -> None:
        if request.url != uri and not request.url.startswith(("data:", "blob:")):
            errors.append(f"external asset: {request.url}")

    page.on("request", record_request)
    page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


def check_guide(guide: Path, chromium: str | None, screenshots: Path | None) -> None:
    """Check a local guide using the starter's language and figure controls.

    Args:
        guide: HTML file to open directly from disk.
        chromium: Chromium executable, or None for Playwright's browser.
        screenshots: Optional output directory for English and Chinese previews.

    Raises:
        ValueError: A rendered state, layout, dependency or link check failed.
    """
    uri = guide.resolve(strict=True).as_uri()
    with sync_playwright() as api:
        browser = api.chromium.launch(executable_path=chromium)
        for width in (390, 1280):
            for scripting in (True, False):
                context = browser.new_context(
                    viewport={"width": width, "height": 850},
                    offline=True,
                    java_script_enabled=scripting,
                    reduced_motion="reduce",
                )
                page = context.new_page()
                errors = _watch_page(page, uri)
                page.goto(uri)
                page.locator("main details").evaluate_all(
                    "items => items.forEach(el => el.open = true)"
                )
                for language in ("en", "zh-CN"):
                    page.locator(f'.language input[value="{language}"]').press("Space")
                    other = "en" if language == "zh-CN" else "zh-CN"
                    _check(
                        page.locator(f'main [lang="{language}"]:visible').count() > 0,
                        f"missing {language} content",
                    )
                    _check(
                        not page.locator(f'main [lang="{other}"]:visible').count(),
                        f"unexpected {other} content",
                    )
                    _check_layout(page)
                    preview = (
                        screenshots / f"{width}-{language}"
                        if scripting and screenshots
                        else None
                    )
                    if preview:
                        preview.mkdir(parents=True, exist_ok=True)
                    _check_figures(page, language, scripting, preview)
                _check(not errors, "\n".join(errors))
                context.close()
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("--chromium", help="Use an existing Chromium executable")
    parser.add_argument(
        "--screenshots", type=Path, help="Directory for rendered figures"
    )
    args = parser.parse_args()
    try:
        check_guide(args.guide, args.chromium, args.screenshots)
    except ValueError as error:
        parser.exit(1, f"{args.guide}: {error}\n")
    print("Passed offline at 390 and 1280 px in both languages with JS on and off.")

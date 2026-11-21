# SPDX-License-Identifier: Apache-2.0
"""Check discovery-supplied management values with a real browser DOM."""

# Third Party
import pytest

# First Party
from tests.import_utils import lmcache_path


@pytest.mark.parametrize(
    "name",
    [
        '<img src=x onerror="window.injected=true">',
        'x" onmouseover="window.injected=true',
        "quotes' & <tags> / 雪",
    ],
)
def test_discovered_node_is_literal_text(name: str) -> None:
    """Render external fields literally and retain exact action identifiers."""
    playwright = pytest.importorskip("playwright.sync_api")
    source = lmcache_path("lmcache_frontend", "static", "js", "app.js").read_text()
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page()
            page.set_content(
                '<table><tbody id="nodeListBody"></tbody></table>'
                + "".join(
                    f'<input id="{field}">'
                    for field in (
                        "nodeName",
                        "nodeHost",
                        "nodePort",
                        "isProxyCheck",
                        "proxyIdInput",
                    )
                )
            )
            page.add_script_tag(content=source)
            page.evaluate(
                """name => {
                    window.injected = false;
                    window.deleted = null;
                    window.alert = message => { throw new Error(message); };
                    window.confirm = () => true;
                    deleteNode = async name => { window.deleted = name; };
                    window.fetch = async () => ({json: async () => ({nodes: [{
                        name: 'proxy', host: 'localhost', port: 8000,
                        is_proxy: true, children: [{
                            name, host: '<svg onload="window.injected=true">',
                            port: '<img src=x>', proxy_id: 'parent" <b> &',
                            is_proxy: false,
                        }],
                    }]})});
                }""",
                name,
            )
            page.evaluate("loadNodeListForManagement()")
            row = page.locator("#nodeListBody tr").nth(1)
            assert row.locator("td").all_text_contents()[:5] == [
                name,
                '<svg onload="window.injected=true">',
                "<img src=x>",
                "No",
                'parent" <b> &',
            ]
            assert row.locator("img, svg, b, script").count() == 0
            assert row.locator("[onerror], [onload], [onmouseover]").count() == 0
            for action in ("edit", "delete"):
                button = row.locator(f".{action}-node")
                assert button.get_attribute("data-name") == name
                button.click()
            assert page.locator("#nodeName").input_value() == name
            assert page.locator("#nodeHost").input_value() == (
                '<svg onload="window.injected=true">'
            )
            assert page.evaluate("window.deleted") == name
            assert page.evaluate("window.injected") is False
        finally:
            browser.close()

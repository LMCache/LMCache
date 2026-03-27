# SPDX-License-Identifier: Apache-2.0
"""Interactive configuration flow for ``lmcache bench engine``.

Entry point: ``run_interactive(args)`` — walks the user through
missing configuration, offers gates for optional settings, and
returns a complete ``argparse.Namespace`` ready for the orchestrator.
"""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.bench.engine_bench.interactive.schema import (
    ConfigItem,
)
from lmcache.cli.commands.bench.engine_bench.interactive.state import (
    InteractiveState,
)
from lmcache.cli.commands.bench.engine_bench.interactive.terminal import (
    BOLD,
    CYAN,
    RESET,
    YELLOW,
    prompt_bool,
    prompt_choice,
    prompt_number,
    prompt_text,
)

__all__ = ["run_interactive"]


# ---------------------------------------------------------------------------
# Prompt dispatcher
# ---------------------------------------------------------------------------


def _prompt_for_item(item: ConfigItem) -> object:
    """Prompt the user for a single config item based on its type."""
    if item.input_type == "text":
        return prompt_text(
            item.display_name,
            item.description,
            default=item.default if item.default is not None else "",
        )
    if item.input_type == "int":
        return prompt_number(
            item.display_name,
            item.description,
            default=item.default,
            number_type=int,
        )
    if item.input_type == "float":
        return prompt_number(
            item.display_name,
            item.description,
            default=item.default,
            number_type=float,
        )
    if item.input_type == "bool":
        return prompt_bool(
            item.display_name,
            item.description,
            default=bool(item.default) if item.default is not None else True,
        )
    if item.input_type == "choice":
        return prompt_choice(
            item.display_name,
            item.description,
            choices=item.choices,
            default=item.default if item.default is not None else "",
        )
    raise ValueError(f"Unknown input_type {item.input_type!r} for {item.key}")


# ---------------------------------------------------------------------------
# Gate prompt
# ---------------------------------------------------------------------------


def _prompt_gate(section_name: str, detail: str) -> bool:
    """Ask the user whether to configure a section or skip with defaults.

    Returns True if the user wants to configure.
    """
    return (
        prompt_choice(
            section_name,
            f"Would you like to configure {detail}?\n"
            f"  Defaults will be used if you skip.",
            choices=[
                ("use defaults", "Skip, use defaults"),
                ("configure", "Yes, configure"),
            ],
            default="use defaults",
        )
        == "configure"
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(state: InteractiveState) -> None:
    """Print a formatted configuration summary."""
    print()
    print(f"{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD} Configuration Summary{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")
    for label, value in state.summary_lines():
        padding = max(0, 26 - len(label))
        print(f"  {label}:{' ' * padding}{CYAN}{value}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


# ---------------------------------------------------------------------------
# Action prompt
# ---------------------------------------------------------------------------


def _prompt_action() -> str:
    """Ask the user to start the benchmark or export config.

    Returns ``"start"`` or ``"export"``.
    """
    return prompt_choice(
        "What would you like to do?",
        "",
        choices=[
            ("start", "Start benchmark"),
            ("export", "Export configuration for later use and exit"),
        ],
        default="start",
    )


def _resolve_before_export(state: InteractiveState) -> None:
    """Resolve tokens_per_gb_kvcache and model before exporting.

    If the user provided an LMCache URL, query the server to get
    ``tokens_per_gb_kvcache`` so the exported config is standalone.
    If the model is empty and an engine URL is available, auto-detect it.
    """
    # First Party
    from lmcache.cli.commands.bench.engine_bench.config import (
        auto_detect_model,
        resolve_tokens_per_gb,
    )

    engine_url = state.get("engine_url", "")
    model = state.get("model", "")

    # Auto-detect model if empty
    if not model and engine_url:
        try:
            model = auto_detect_model(engine_url)
            state.set("model", model)
        except RuntimeError as e:
            print(f"  {YELLOW}Warning: could not auto-detect model: {e}{RESET}")

    # Resolve tokens_per_gb from LMCache if needed
    lmcache_url = state.get("lmcache_url", "")
    if lmcache_url and not state.is_set("tokens_per_gb_kvcache"):
        try:
            tokens = resolve_tokens_per_gb(lmcache_url, model)
            state.set("tokens_per_gb_kvcache", tokens)
        except RuntimeError as e:
            print(
                f"  {YELLOW}Warning: could not resolve "
                f"tokens_per_gb from LMCache: {e}{RESET}"
            )


def _handle_export(state: InteractiveState) -> None:
    """Prompt for filename, resolve values, save JSON, and exit."""
    _resolve_before_export(state)
    filename = prompt_text(
        "Export filename",
        "",
        default="bench_config.json",
    )
    state.save_json(filename)
    print()
    print(f"  {CYAN}Saved to {filename}{RESET}")
    print(
        f"  {BOLD}Replay with:{RESET} "
        f"{CYAN}lmcache bench engine "
        f"--engine-url <URL> --config {filename}{RESET}"
    )
    print()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_interactive(args: argparse.Namespace) -> argparse.Namespace:
    """Run the interactive configuration flow.

    Walks the user through missing required items, offers gates for
    general and workload-specific settings, shows a summary, and
    returns a complete ``argparse.Namespace``.

    Args:
        args: Partially-populated CLI args (some may be None).

    Returns:
        A fully-populated ``argparse.Namespace`` ready for the
        benchmark orchestrator.
    """
    state = InteractiveState.from_cli_args(args)

    print()
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD} lmcache bench engine — Interactive Setup{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    # ── Phase 1: Required items ───────────────────────────────────────
    # Walk through missing required items one by one.  Re-evaluate the
    # list after each prompt because setting one value (e.g., has_lmcache)
    # can make new items eligible (e.g., lmcache_url) or skip others
    # (e.g., tokens_per_gb_kvcache).
    while True:
        missing = state.get_missing_required()
        if not missing:
            break
        item = missing[0]
        value = _prompt_for_item(item)
        state.set(item.key, value)

    # ── Phase 2: General settings gate ────────────────────────────────
    if state.has_unconfigured_general():
        if _prompt_gate(
            "General settings",
            "general settings (model, KV cache volume, etc.)",
        ):
            for item in state.get_general_items():
                value = _prompt_for_item(item)
                state.set(item.key, value)

    # ── Phase 3: Workload settings gate (always shown) ────────────────
    if state.has_workload_items():
        workload = state.get("workload", "workload")
        if _prompt_gate(
            f"Workload settings ({workload})",
            "workload-specific settings",
        ):
            for item in state.get_workload_items():
                value = _prompt_for_item(item)
                state.set(item.key, value)

    # Fill all remaining defaults
    state.fill_defaults()

    # ── Phase 4: Summary + action ─────────────────────────────────────
    _print_summary(state)
    action = _prompt_action()

    if action == "export":
        _handle_export(state)

    # Carry over output settings from original CLI args
    ns = state.to_namespace()
    for attr in ("output_dir", "seed", "no_csv", "json", "quiet", "format", "output"):
        cli_val = getattr(args, attr, None)
        if cli_val is not None:
            setattr(ns, attr, cli_val)

    return ns

# SPDX-License-Identifier: Apache-2.0
"""Read manifest.toml, the scope of one lmcache-client build.

    python tools/client_wheel/scope.py [--manifest M]

A tracked file under lmcache/ is shipped if it is a compat shim or, with
--boundary transitional, a transitional file. Otherwise it is shipped if it
matches trees or files and no [[exclude]] entry: an exclusion always wins.
As a script it prints a summary of the scope.
"""

# Standard
from pathlib import Path
import argparse
import sys

if sys.version_info >= (3, 11):
    # Standard
    import tomllib
else:
    # Third Party
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
KEYS = (
    "trees",
    "files",
    "deny",
    "build_inputs",
    "native_allow",
    "compat_shims",
    "exclude",
    "transitional",
)


def matches(path: str, rules: list[str]) -> bool:
    return any(path == r or (r.endswith("/") and path.startswith(r)) for r in rules)


class Scope:
    """The manifest of one build and the rules that select its files."""

    def __init__(self, manifest: dict, path: Path) -> None:
        self.manifest = manifest
        self.path = path

    @property
    def rules(self) -> list[str]:
        return self.manifest["trees"] + self.manifest["files"]

    @property
    def shim_paths(self) -> list[str]:
        return [shim["path"] for shim in self.manifest.get("compat_shims", [])]

    @property
    def transitional_paths(self) -> list[str]:
        groups = self.manifest.get("transitional", [])
        return [path for group in groups for path in group["paths"]]

    @property
    def exclude_paths(self) -> list[str]:
        groups = self.manifest.get("exclude", [])
        return [path for group in groups for path in group["paths"]]

    def excluded(self, path: str) -> dict | None:
        """The [[exclude]] entry that matches ``path``, if any."""
        groups = self.manifest.get("exclude", [])
        return next((group for group in groups if matches(path, group["paths"])), None)

    def allowed(self, boundary: str) -> list[str]:
        """Deny-area files this build may ship."""
        transitional = self.transitional_paths if boundary == "transitional" else []
        return self.shim_paths + transitional

    def selects(self, path: str, boundary: str) -> bool:
        """Whether build.py stages the tracked ``path`` under lmcache/."""
        if path in self.allowed(boundary):
            return True
        return self.excluded(path) is None and matches(path, self.rules)

    def summary(self) -> list[str]:
        lines = [f"include rules: {len(self.rules)}"]
        for group in self.manifest.get("exclude", []):
            lines.append(f"exclude {group['owner']}: {len(group['paths'])} paths")
        groups = self.manifest.get("transitional", [])
        return lines + [f"transitional groups: {len(groups)}"]


def validate(scope: Scope) -> list[str]:
    return [f"unknown key: {key}" for key in sorted(set(scope.manifest) - set(KEYS))]


def load_scope(path: Path | None = None) -> Scope:
    """Read manifest.toml and exit if it has a key no tool reads."""
    path = path or HERE / "manifest.toml"
    scope = Scope(tomllib.loads(path.read_text()), path)
    errors = validate(scope)
    if errors:
        raise SystemExit(f"{path.name}:\n  " + "\n  ".join(errors))
    return scope


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print("\n".join(load_scope(args.manifest).summary()))


if __name__ == "__main__":
    main()

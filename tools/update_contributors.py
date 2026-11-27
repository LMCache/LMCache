# SPDX-License-Identifier: Apache-2.0
"""Keep contributor credit for LMCache in sync with GitHub.

Subcommands:

* ``contributors`` regenerates CONTRIBUTORS.md: every GitHub account with a
  commit on the default branch, listed by handle.
* ``thanks BASE HEAD`` prints a "Thanks" section for release notes, covering
  everyone with a commit between the two refs.
* ``check`` validates ACKNOWLEDGMENTS.md.

The project never infers an affiliation from email domains, GitHub profiles,
or anything else. A name or affiliation appears only when the contributor
added it to ACKNOWLEDGMENTS.md themselves. Everyone else is credited by
GitHub handle.

Only the Python standard library is used. Set GITHUB_TOKEN to raise the
GitHub API rate limit.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import argparse
import json
import os
import re
import sys
import urllib.request

REPO = "LMCache/LMCache"
API_URL = "https://api.github.com"
ROOT = Path(__file__).resolve().parent.parent
CONTRIBUTORS_FILE = ROOT / "CONTRIBUTORS.md"
ACKNOWLEDGMENTS_FILE = ROOT / "ACKNOWLEDGMENTS.md"
TABLE_START = "<!-- acknowledgments:start -->"
TABLE_END = "<!-- acknowledgments:end -->"

_HANDLE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
_SEPARATOR_RE = re.compile(r"^:?-+:?$")

CONTRIBUTORS_HEADER = """\
# Contributors

Thank you to everyone who has contributed to LMCache.

This file lists every GitHub account with a commit on the `dev` branch, by
GitHub handle, in alphabetical order. It is generated, so please do not edit
it by hand. To regenerate it, run:

```bash
python tools/update_contributors.py contributors
```

To choose the name and affiliation shown for you in release notes, add
yourself to [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).

If you contributed and are not listed, the email address on your commits is
probably not linked to your GitHub account. See GitHub's guide to
[setting your commit email address](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address).
"""


@dataclass(frozen=True)
class Acknowledgment:
    """One contributor's self-chosen credit.

    Attributes:
        handle: GitHub handle, without the leading "@".
        name: Display name, or an empty string to use the handle only.
        affiliation: Affiliation to credit, or an empty string for none.
    """

    handle: str
    name: str = ""
    affiliation: str = ""


def is_bot(login: str, account_type: str = "User") -> bool:
    """Return True if a GitHub account is an automation account.

    Args:
        login: The GitHub login.
        account_type: The "type" field GitHub reports for the account.

    Returns:
        True for bots and apps, False for people.
    """
    return account_type == "Bot" or login.lower().endswith("[bot]")


def _normalize_handle(cell: str) -> str:
    """Strip markup from a handle cell and validate it.

    Args:
        cell: The raw table cell, such as "@quaid" or "`@quaid`".

    Returns:
        The bare handle.

    Raises:
        ValueError: If the cell is not a valid GitHub handle.
    """
    handle = cell.strip().strip("`").lstrip("@")
    if not _HANDLE_RE.match(handle):
        raise ValueError(f"not a valid GitHub handle: {cell!r}")
    return handle


def parse_acknowledgments(text: str) -> Dict[str, Acknowledgment]:
    """Parse the table in ACKNOWLEDGMENTS.md.

    Only rows between the start and end markers are read. The first three
    columns are handle, name, and affiliation; later columns are ignored.

    Args:
        text: The full contents of ACKNOWLEDGMENTS.md.

    Returns:
        A mapping from lowercased handle to that contributor's entry.

    Raises:
        ValueError: If the markers are missing, a handle is invalid, or a
            handle appears twice.
    """
    try:
        start = text.index(TABLE_START) + len(TABLE_START)
        end = text.index(TABLE_END, start)
    except ValueError as err:
        raise ValueError("acknowledgments table markers not found") from err

    entries: Dict[str, Acknowledgment] = {}
    for line in text[start:end].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(_SEPARATOR_RE.match(cell) for cell in cells):
            continue
        if cells[0].lower() == "github handle":
            continue
        cells += [""] * (3 - len(cells))
        handle = _normalize_handle(cells[0])
        key = handle.lower()
        if key in entries:
            raise ValueError(f"duplicate row for @{handle}")
        entries[key] = Acknowledgment(handle, cells[1], cells[2])
    return entries


def _sorted_handles(handles: Iterable[str]) -> List[str]:
    """Deduplicate handles case-insensitively and sort them alphabetically."""
    unique: Dict[str, str] = {}
    for handle in handles:
        unique.setdefault(handle.lower(), handle)
    return [unique[key] for key in sorted(unique)]


def render_contributors(handles: Iterable[str]) -> str:
    """Render CONTRIBUTORS.md.

    Args:
        handles: GitHub handles of every contributor. Bots should already be
            removed.

    Returns:
        The full file contents.
    """
    lines = [CONTRIBUTORS_HEADER, "## Everyone", ""]
    lines += [f"- [@{h}](https://github.com/{h})" for h in _sorted_handles(handles)]
    return "\n".join(lines) + "\n"


def _credit(handle: str, entries: Dict[str, Acknowledgment]) -> str:
    """Format one person as they asked to be credited."""
    entry = entries.get(handle.lower())
    if entry and entry.name:
        return f"{entry.name} (@{handle})"
    return f"@{handle}"


def render_thanks(handles: Iterable[str], entries: Dict[str, Acknowledgment]) -> str:
    """Render the release-notes "Thanks" section.

    Contributors who listed an affiliation are grouped under it, groups in
    alphabetical order. Everyone else is listed by handle in a final line.

    Args:
        handles: GitHub handles of everyone who contributed to the release.
        entries: Parsed ACKNOWLEDGMENTS.md.

    Returns:
        Markdown for the section, ending with a newline.
    """
    people = _sorted_handles(handles)
    groups: Dict[str, List[str]] = {}
    others: List[str] = []
    for handle in people:
        entry = entries.get(handle.lower())
        if entry and entry.affiliation:
            groups.setdefault(entry.affiliation, []).append(handle)
        else:
            others.append(handle)

    noun = "person" if len(people) == 1 else "people"
    lines = [
        "## Thanks",
        "",
        f"Thank you to the {len(people)} {noun} who contributed to this release.",
        "",
    ]
    for affiliation in sorted(groups, key=str.lower):
        credits = ", ".join(_credit(h, entries) for h in groups[affiliation])
        lines.append(f"- {affiliation}: {credits}")
    if others:
        lines.append("- " + ", ".join(_credit(h, entries) for h in others))
    lines += [
        "",
        "Affiliations are self-reported in "
        f"[ACKNOWLEDGMENTS.md](https://github.com/{REPO}/blob/dev/ACKNOWLEDGMENTS.md).",
    ]
    return "\n".join(lines) + "\n"


def _get_json(url: str) -> Any:
    """GET a GitHub API URL and decode the JSON body."""
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "lmcache-update-contributors")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _paginate(url: str, key: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield items from every page of a GitHub list endpoint."""
    separator = "&" if "?" in url else "?"
    page = 1
    while True:
        body = _get_json(f"{url}{separator}per_page=100&page={page}")
        items = body[key] if key else body
        yield from items
        if len(items) < 100:
            return
        page += 1


def fetch_contributors(repo: str = REPO) -> List[str]:
    """Return the handles of every human with a commit on the default branch.

    Args:
        repo: The "owner/name" of the repository.

    Returns:
        GitHub handles, bots excluded.
    """
    return [
        user["login"]
        for user in _paginate(f"{API_URL}/repos/{repo}/contributors")
        if not is_bot(user["login"], user.get("type", "User"))
    ]


def fetch_release_authors(
    base: str, head: str, repo: str = REPO
) -> Tuple[List[str], int]:
    """Return the handles of everyone with a commit between two refs.

    Args:
        base: The previous release tag or ref.
        head: The new release tag or ref.
        repo: The "owner/name" of the repository.

    Returns:
        A tuple of (handles, unlinked), where unlinked counts commits whose
        email address is not linked to any GitHub account.
    """
    handles: List[str] = []
    unlinked = 0
    url = f"{API_URL}/repos/{repo}/compare/{base}...{head}"
    for commit in _paginate(url, key="commits"):
        author = commit.get("author")
        if not author:
            unlinked += 1
        elif not is_bot(author["login"], author.get("type", "User")):
            handles.append(author["login"])
    return handles, unlinked


def _read_acknowledgments() -> Dict[str, Acknowledgment]:
    return parse_acknowledgments(ACKNOWLEDGMENTS_FILE.read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments, excluding the program name. Defaults to sys.argv.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    contributors = sub.add_parser("contributors", help="regenerate CONTRIBUTORS.md")
    contributors.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if CONTRIBUTORS.md is out of date instead of writing it",
    )
    thanks = sub.add_parser("thanks", help="print a release-notes Thanks section")
    thanks.add_argument("base", help="previous release tag, such as v0.5.7")
    thanks.add_argument("head", help="new release tag or branch")
    sub.add_parser("check", help="validate ACKNOWLEDGMENTS.md")
    args = parser.parse_args(argv)

    try:
        entries = _read_acknowledgments()
    except ValueError as err:
        print(f"ACKNOWLEDGMENTS.md: {err}", file=sys.stderr)
        return 1

    if args.command == "check":
        print(f"ACKNOWLEDGMENTS.md: {len(entries)} entries, OK")
        return 0

    if args.command == "thanks":
        handles, unlinked = fetch_release_authors(args.base, args.head)
        sys.stdout.write(render_thanks(handles, entries))
        if unlinked:
            print(
                f"note: {unlinked} commits have no linked GitHub account "
                "and are not credited above",
                file=sys.stderr,
            )
        return 0

    rendered = render_contributors(fetch_contributors())
    current = (
        CONTRIBUTORS_FILE.read_text(encoding="utf-8")
        if CONTRIBUTORS_FILE.exists()
        else ""
    )
    if args.check:
        if current != rendered:
            print(
                "CONTRIBUTORS.md is out of date; run without --check", file=sys.stderr
            )
            return 1
        return 0
    CONTRIBUTORS_FILE.write_text(rendered, encoding="utf-8")
    print(f"wrote {CONTRIBUTORS_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

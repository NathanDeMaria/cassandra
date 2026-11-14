"""Move the git pins in pyproject.toml to what their upstreams have merged.

    python bump_deps.py list            # upstream repositories with a newer commit
    python bump_deps.py apply EndGame   # rewrite that repository's pins in place

Every dependency under `[tool.uv.sources]` is pinned to a commit, on purpose:
a bump moves the data or the fit a replay reads, so it should arrive as a
diff somebody merges. What that left by hand was the mechanical half --
notice the upstream merged, look up the sha, edit the line, `uv lock`, open
the pull request. The Pins workflow runs this on a schedule and does that
half; reading the pull request and merging it stays a person's.

Where a pin goes:

- A package that is the whole repository goes to the default branch's head.
- A package in a `subdirectory` goes to the last commit that touched that
  directory, so a change elsewhere in the repository is not a bump.
- A package that another pin from the same repository pins itself follows
  that pin. `endgame-aws` names the `endgame` rev it was built against in its
  own `[tool.uv.sources]`, and the two must agree about what a Game is, so
  `endgame` moves only when `endgame-aws` says so.
- A pin that is already ahead of the default branch -- a commit from a pull
  request not merged yet -- is left alone rather than moved backwards.

`apply` prints the pull request's title, a blank line, and its body. It does
not run `uv lock`; the workflow does, and a lock that fails is the report
that the upstreams disagree.
"""

import argparse
import base64
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PYPROJECT = Path(__file__).parent / "pyproject.toml"

# At most this many upstream commit subjects under one pin in the body
MAX_SUBJECTS = 15

# A GitHub API path to its decoded JSON, or None for a 404
Get = Callable[[str], Any]


@dataclass(frozen=True)
class Pin:
    name: str
    repo: str  # "owner/name"
    rev: str
    subdirectory: str | None = None

    @property
    def group(self) -> str:
        """The repository's name: what a pull request is opened per."""
        return self.repo.split("/")[1]


@dataclass(frozen=True)
class Bump:
    pin: Pin
    rev: str
    follows: str | None = None  # the pin whose own pyproject chose this rev


def github(path: str) -> Any:
    """GET one GitHub API path. A token only raises the rate limit."""
    request = urllib.request.Request(f"https://api.github.com/{path}")
    request.add_header("Accept", "application/vnd.github+json")
    if token := os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def read_pins(pyproject: str) -> list[Pin]:
    """The `[tool.uv.sources]` entries pinned to a commit of a GitHub repository."""
    sources = tomllib.loads(pyproject).get("tool", {}).get("uv", {}).get("sources", {})
    pins = []
    for name, source in sources.items():
        if not isinstance(source, dict) or "rev" not in source:
            continue
        match = re.fullmatch(
            r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?/?", source.get("git", "")
        )
        if match:
            pins.append(
                Pin(name, match.group(1), source["rev"], source.get("subdirectory"))
            )
    return pins


def _latest(pin: Pin, get: Get) -> str:
    path = f"repos/{pin.repo}/commits?per_page=1"
    if pin.subdirectory:
        path += f"&path={pin.subdirectory}"
    return get(path)[0]["sha"]


def _own_pins(pin: Pin, rev: str, get: Get) -> list[Pin]:
    """What the package pins for itself at `rev`."""
    directory = f"{pin.subdirectory}/" if pin.subdirectory else ""
    found = get(f"repos/{pin.repo}/contents/{directory}pyproject.toml?ref={rev}")
    if found is None:
        return []
    return read_pins(base64.b64decode(found["content"]).decode())


def plan(pyproject: str, get: Get = github) -> list[Bump]:
    """Every pin that should move, in the order pyproject.toml lists them."""
    pins = read_pins(pyproject)
    targets = {pin.name: _latest(pin, get) for pin in pins}
    follows: dict[str, str] = {}
    for pin in pins:
        if pin.name in follows:
            continue  # its own rev is somebody else's choice; so is what it pins
        for own in _own_pins(pin, targets[pin.name], get):
            if own.repo == pin.repo and own.name in targets and own.name != pin.name:
                targets[own.name] = own.rev
                follows[own.name] = pin.name
    bumps = []
    for pin in pins:
        rev = targets[pin.name]
        if rev == pin.rev:
            continue
        compared = get(f"repos/{pin.repo}/compare/{rev}...{pin.rev}")
        if compared is not None and compared.get("status") == "ahead":
            continue  # the pin is past the target already: an unmerged commit
        bumps.append(Bump(pin, rev, follows.get(pin.name)))
    return bumps


def apply(pyproject: str, bumps: list[Bump]) -> str:
    """`pyproject` with each bumped pin's rev replaced, and nothing else touched."""
    for bump in bumps:
        line = re.compile(
            rf'^({re.escape(bump.pin.name)} = \{{[^\n]*\brev = "){bump.pin.rev}(")',
            re.MULTILINE,
        )
        pyproject, count = line.subn(rf"\g<1>{bump.rev}\g<2>", pyproject)
        if count != 1:
            raise ValueError(
                f"expected one source line for {bump.pin.name}, found {count}"
            )
    return pyproject


def _subjects(bump: Bump, get: Get) -> list[str]:
    compared = get(f"repos/{bump.pin.repo}/compare/{bump.pin.rev}...{bump.rev}")
    commits = (compared or {}).get("commits", [])
    subjects = [commit["commit"]["message"].splitlines()[0] for commit in commits]
    if len(subjects) > MAX_SUBJECTS:
        more = len(subjects) - MAX_SUBJECTS
        subjects = subjects[:MAX_SUBJECTS] + [f"... and {more} more"]
    return subjects


def describe(bumps: list[Bump], get: Get = github) -> tuple[str, str]:
    """The pull request's title and body for one repository's bumps."""
    names = [bump.pin.name for bump in bumps]
    title = "Bump " + (
        names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    )
    repo = bumps[0].pin.repo
    lines = [f"[{repo}](https://github.com/{repo}) has merged past the pin.", ""]
    for bump in bumps:
        old, new = bump.pin.rev, bump.rev
        compare = f"https://github.com/{repo}/compare/{old}...{new}"
        entry = f"- `{bump.pin.name}` {old[:7]} -> {new[:7]} ([compare]({compare}))"
        if bump.follows:
            entry += f", the rev `{bump.follows}` pins it at"
        lines.append(entry)
        lines += [f"  - {subject}" for subject in _subjects(bump, get)]
    lines += [
        "",
        "Opened by the Pins workflow, which rewrites this branch when main or "
        "the upstream moves. `uv.lock` is re-resolved; nothing else changes.",
    ]
    return title, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="upstream repositories with a newer commit")
    applying = commands.add_parser("apply", help="rewrite one repository's pins")
    applying.add_argument("group", help="a repository name printed by `list`")
    args = parser.parse_args(argv)

    pyproject = PYPROJECT.read_text()
    bumps = plan(pyproject)
    if args.command == "list":
        for group in dict.fromkeys(bump.pin.group for bump in bumps):
            print(group)
        return 0
    bumps = [bump for bump in bumps if bump.pin.group == args.group]
    if not bumps:
        print(f"nothing to bump for {args.group}", file=sys.stderr)
        return 1
    PYPROJECT.write_text(apply(pyproject, bumps))
    title, body = describe(bumps)
    print(f"{title}\n\n{body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

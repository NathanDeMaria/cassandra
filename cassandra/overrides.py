"""Team news `picks.py` keeps applying until it expires.

`--qb-out TEAM` and `--exclude TEAM` last one run, and some news lasts a
season: a starter out for the year has to be passed again every week,
since the quarterback index hands a missed start on for one game only.
A file of overrides carries them between runs, each with the date it
stops applying and a note on why:

    [[qb_out]]
    team = "UCLA Bruins"
    until = 2026-11-28
    note = "Iamaleava out for the season"

    [[exclude]]
    team = "Syracuse Orange"
    until = 2026-11-01
    note = "QB crisis; look again"

`until` is required, so a stale injury can't outlive the season unread. A
run prints every override it applied and every one that has expired.
"""

from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import NamedTuple

import tomllib

KINDS = ("qb_out", "exclude")


class Override(NamedTuple):
    kind: str
    team: str
    until: date
    note: str


def parse(text: str, source: str = "overrides") -> list[Override]:
    """The overrides in a TOML file's text. Anything malformed is an error."""
    found: list[Override] = []
    data = tomllib.loads(text)
    unknown = sorted(set(data) - set(KINDS))
    if unknown:
        raise ValueError(f"{source}: unknown sections {unknown}; known: {KINDS}")
    for kind in KINDS:
        for entry in data.get(kind, []):
            team, until = entry.get("team"), entry.get("until")
            if not isinstance(team, str) or not team:
                raise ValueError(f"{source}: a [[{kind}]] entry has no team")
            if not isinstance(until, date):
                raise ValueError(
                    f"{source}: [[{kind}]] {team!r} needs an until date, e.g. "
                    "until = 2026-11-28"
                )
            found.append(Override(kind, team, until, str(entry.get("note", ""))))
    return found


def load(path: Path) -> list[Override]:
    """A file's overrides; none when it doesn't exist."""
    if not path.exists():
        return []
    return parse(path.read_text(), str(path))


def split(
    overrides: Iterable[Override], today: date
) -> tuple[list[Override], list[Override]]:
    """The overrides still applying on `today` (through `until`), and the expired."""
    active, expired = [], []
    for override in overrides:
        (active if today <= override.until else expired).append(override)
    return active, expired


def teams(overrides: Sequence[Override], kind: str) -> list[str]:
    return [o.team for o in overrides if o.kind == kind]

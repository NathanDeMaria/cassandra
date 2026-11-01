"""Who has been starting at quarterback this season, and whether that man is
in doubt for his team's next game.

The availability index (`cassandra.predictor.qb_out`) answers a question
about games already played: was the starter missing from this one? A bet
needs the next game, which nobody has plays for. What the last game says
about it is a lot. Going into a game, the man with the most starts this
season (three or more; ncaafb 2014-2026, 24,623 team-games) failed to start:

    after a game he played through        8%
    after one he missed entirely          79%
    after one he came off the bench in    68%
    after one he left early, not a rout   60%
    after one he left in a rout           34%

The 8% is midweek news no play can carry. The rest is what this reads.

It matters to a pick backing that team. The market reads the injury report
and the model doesn't: of the Kalshi picks the live rule would have made
(2025-26, edge 7+, 10+ at 80c and up), the 64 backing a team whose starter
missed, left early or came off the bench were worth +2.5% at the close
against +9.0% for the other 374 (edge 5+: +0.7% against +4.1%, 119 and
729). Where the backed team's starter then didn't start, +0.9% at the close
and +0.6% realized, against +4.5% and +14.5%. The 109 picks against such a
team were worth +4.6% at the close, the others' rate, so only the backed
side is in doubt. Kept that way through a whole absence: the picks were
worth no more at the close with the backup on his third straight start
than on his first.

`{league}_qb_starts.json` holds the season being played, game by game, as
the quarterback sweep reads it (`cassandra.qb_out_build.build_all`).
`picks.py` adds tonight's games read off ESPN, and asks `doubts` about every
team with a game coming up.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

from endgame.types import Game
from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"
#: Game days are named in Central time, as the picks' emails are.
_CENTRAL = ZoneInfo("America/Chicago")

#: Under this share of his team's attempts, he didn't play the game out.
LEFT_SHARE = 0.6
#: ...if somebody else threw at least this many: not a run-heavy afternoon.
MIN_COVER_ATTEMPTS = 5
#: A lead or deficit at his last snap that reads as resting him, from the
#: third quarter on. 34% of those starters missed the next game, against 60%
#: who left anything closer.
ROUT = 21


def qb_starts_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_qb_starts.json"


class GameQb(BaseModel):
    """One team's quarterbacks in one game, by `say_youll_remember_me.qb`'s
    name keys."""

    #: Who threw the team's early passes.
    starter: str
    #: Passes thrown, by man.
    attempts: dict[str, int]
    #: Display names, by key, as the play text writes them.
    names: dict[str, str]
    #: Everyone credited with a pass, a run or a sack.
    snaps: list[str]
    #: Per man who took a snap: the period of his last one, and his team's
    #: lead then (negative behind).
    last_snap: dict[str, tuple[int | None, int | None]]


class QbStartsFile(BaseModel):
    """The `{league}_qb_starts.json` artifact: the latest season's games."""

    league: str
    season: int
    #: game id -> team (the canonical name) -> its quarterbacks.
    games: dict[str, dict[str, GameQb]]


def read_qb_starts(league: str) -> QbStartsFile | None:
    path = qb_starts_path(league)
    if not path.exists():
        return None
    return QbStartsFile.model_validate_json(path.read_text())


def write_qb_starts(starts: QbStartsFile) -> Path:
    path = qb_starts_path(starts.league)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(starts.model_dump_json())
    return path


class Doubt(NamedTuple):
    """A team whose starter's last game says he may not start its next."""

    team: str
    #: missed / left early / came off the bench
    kind: str
    #: For a person: who, how many starts, what happened, and when.
    sentence: str


def _period(period: int | None) -> str:
    if period is None:
        return "?"
    if period >= 5:
        return "overtime"
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(period, f"{period}th") + " quarter"


def doubt(team: str, played: Sequence[tuple[Game, GameQb]]) -> Doubt | None:
    """Whether the man with the most starts in `played` (the team's games
    with passing, in order) is in doubt after the last of them.

    Ties in starts go to the earlier first start, as in the availability
    index. The doubt lasts until the man covering has more starts than him
    or he plays a game through.
    """
    if not played:
        return None
    starts: Counter[str] = Counter()
    first: dict[str, int] = {}
    names: dict[str, str] = {}
    for i, (_, qb) in enumerate(played):
        starts[qb.starter] += 1
        first.setdefault(qb.starter, i)
        names.update(qb.names)
    est = max(starts, key=lambda k: (starts[k], -first[k]))
    game, last = played[-1]
    who = f"{names.get(est, est)} ({starts[est]} start{'s' * (starts[est] != 1)})"
    when = f"{game.date.astimezone(_CENTRAL):%b %d}"
    if est not in last.snaps:
        return Doubt(
            team,
            "missed",
            f"{team}: {who} didn't play {when}; "
            f"{names.get(last.starter, last.starter)} started.",
        )
    thrown = sum(last.attempts.values())
    mine = last.attempts.get(est, 0)
    cover = max(
        ((k, n) for k, n in last.attempts.items() if k != est),
        key=lambda item: item[1],
        default=(None, 0),
    )
    if not thrown or mine / thrown >= LEFT_SHARE or cover[1] < MIN_COVER_ATTEMPTS:
        return None
    other = f"{names.get(cover[0], cover[0])} threw {cover[1]}"
    if last.starter != est:
        return Doubt(
            team,
            "came off the bench",
            f"{team}: {names.get(last.starter, last.starter)} started {when} over "
            f"{who}, who threw {mine} of {thrown} passes.",
        )
    period, lead = last.last_snap.get(est, (None, None))
    if period is not None and lead is not None and period >= 3 and abs(lead) >= ROUT:
        return None
    return Doubt(
        team,
        "left early",
        f"{team}: {who} threw {mine} of {thrown} passes {when}, last snap in the "
        f"{_period(period)}; {other}.",
    )


def doubts(
    games: Iterable[Game], starts: Mapping[str, Mapping[str, GameQb]]
) -> dict[str, Doubt]:
    """Every team in doubt going into its next game, by name.

    `games` is the season's schedule; `starts` the per-game quarterbacks,
    keyed like `QbStartsFile.games`. A game without them (no plays, or no
    passing) is passed over, as the availability index passes it.
    """
    played: dict[str, list[tuple[Game, GameQb]]] = {}
    for game in sorted((g for g in games if g.completed), key=lambda g: g.date):
        for team, qb in starts.get(game.game_id, {}).items():
            played.setdefault(team, []).append((game, qb))
    found = {}
    for team, history in played.items():
        if (d := doubt(team, history)) is not None:
            found[team] = d
    return found

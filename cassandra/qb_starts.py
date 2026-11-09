"""Who has been starting at quarterback this season, and whether that man is
in doubt for his team's next game.

The availability index (`cassandra.predictor.qb_out`) answers a question
about games already played: was the starter missing from this one? A bet
needs the next game, which nobody has plays for. What the last game says
about it is a lot. Going into a game, the starter by the availability
index's rule (`walk`) was out of it, ncaafb 2014-2026 (32,599 team-games):

    after a game he played through        7%
    after one he missed entirely          62%
    after one he left early, not a rout   47%
    after one he came off the bench in    33%

The 7% is midweek news no play can carry. The rest is what this reads, and
what `availability` prices a fixture at (`AVAILABILITY_OUT`). The miss holds
at 59-68% however long he has been out; the index's flag for a long absence
weighs less, and so does the price.

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
than on his first. (Those counts named the starter by most starts this
season, which read a preseason starter back from injury as the backup and
held his picks the week he returned; `walk` is the index's own rule.)

Pricing the doubt rather than holding moved no money: re-picked on the
2025-26 Kalshi books with `availability` in the model, the picks were worth
the same at the close (3.54 against 3.55 on Kelly, 33.7 against 35.3
close-dollars flat over ~440 picks). It makes the published numbers better,
not the bets, so the hold stays.

`{league}_qb_starts.json` holds the season being played, game by game, as
the quarterback sweep reads it (`cassandra.qb_out_build.build_all`).
`picks.py` adds tonight's games read off ESPN, and asks `doubts` and
`availabilities` about every team with a game coming up.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

from endgame.types import Game
from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME
from cassandra.predictor.qb_out import QB_OUT_DECAY

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

#: The leagues `availabilities` prices: the one `AVAILABILITY_OUT` was
#: measured on.
PRICED_LEAGUES: frozenset[str] = frozenset({"ncaafb"})

#: The chance the starter is out of his team's next game, by how his last
#: one went (`Doubt.kind`): ncaafb 2014-2026, through `walk` and `doubt`.
AVAILABILITY_OUT: Mapping[str, float] = {
    "missed": 0.62,
    "left early": 0.47,
    "came off the bench": 0.33,
}


#: Consecutive starts by one man, without the starter, that make him the
#: starter -- per league, because a college quarterback change is usually
#: not an injury. See `cassandra.qb_out_build` for the measurements.
HANDOVER: Mapping[str, int] = {"nfl": 3, "ncaafb": 1}

#: Leagues where the handover waits for the starter to be seen healthy: "A
#: starter nobody has seen since" in `cassandra.qb_out_build`. The NFL's
#: three-game handover was measured on its own and left alone.
KEEP_UNSEEN: frozenset[str] = frozenset({"ncaafb"})

#: Per league, the starts that make a man last season's starter: he carries
#: them all into this season's standing, so if he misses the opener he
#: outranks the man who covered it the day he's back, rather than reading
#: as that man's absence. Nobody else carries anything. Absent where it
#: wasn't measured.
#:
#: ncaafb 2014-2026: holds firing on a team whose last-season starter was
#: the one playing, 228 with no carry -> 2; a capped carry for every man
#: with starts last season left 134 (2 each) to 71 (4 each), and hid real
#: absences of last year's backups. The model can't tell any of them apart
#: (brier within +/-1e-5 of none, live and with every flag known).
CARRY: Mapping[str, int] = {"ncaafb": 6}


class StarterWalk:
    """One team's season, a game at a time, by the availability index's rule
    (`cassandra.qb_out_build`): who the starter is, and whether he's out.

    Here so the index and the doubt below ask about the same man. They used
    to differ: the doubt took whoever had the most starts, which reads a
    preseason starter back from injury as the backup, and held his team's
    picks the week he returned.
    """

    def __init__(
        self,
        handover: int,
        keep_unseen: bool,
        last_season: Mapping[str, int] | None = None,
        carry: int | None = None,
    ) -> None:
        self.handover, self.keep_unseen = handover, keep_unseen
        #: Last season's starter's starts, if he made `carry` of them.
        self._carried: dict[str, int] = {}
        if carry is not None and last_season:
            key = max(last_season, key=lambda k: last_season[k])
            if last_season[key] >= carry:
                self._carried[key] = last_season[key]
        #: The starter, None before the team's first game with passing.
        self.expected: str | None = None
        #: Games he has missed running, 0 when he's playing.
        self.absent = 0
        #: Whether he has taken a snap in a game he didn't start since his
        #: last start: in relief, or on the field -- a benching's look.
        self.seen = False
        self._covering: tuple[str | None, int] = (None, 0)
        self._starts: Counter[str] = Counter()
        self._first: dict[str, int] = {}

    def starts(self, key: str) -> int:
        return self._starts[key]

    def _standing(self, key: str) -> tuple[int, int]:
        return (
            self._starts[key] + self._carried.get(key, 0),
            -self._first.get(key, len(self._first)),
        )

    def _hand_to(self, key: str) -> None:
        self.expected, self.absent, self.seen = key, 0, False
        self._covering = (None, 0)

    def going_in(self, last_starter: str) -> str | None:
        """The starter going into the next game: the man who started the
        last one if he now outranks the starter -- the job is his the moment
        he starts again -- else the starter."""
        expected = self.expected
        if expected is not None and self._standing(last_starter) > self._standing(
            expected
        ):
            return last_starter
        return expected

    def step(self, starter: str, snaps: Iterable[str]) -> int | None:
        """Fold in one game with passing: who started, and who took a snap.
        Which game of the starter's absence it was, or None if he played."""
        out = None
        expected = self.expected
        if expected is None or self._standing(starter) > self._standing(expected):
            # First game, or the established man starting in place of a
            # cover: he has the job, and nobody is missing.
            self._hand_to(starter)
        elif expected in set(snaps):
            self._covering = (None, 0)
            if starter == expected:
                self.absent, self.seen = 0, False
            else:
                self.seen = True
        else:
            self.absent += 1
            out = self.absent
            cover, streak = self._covering
            streak = streak + 1 if cover == starter else 1
            if streak >= self.handover and (not self.keep_unseen or self.seen):
                self._hand_to(starter)
            else:
                self._covering = (starter, streak)
        self._starts[starter] += 1
        self._first.setdefault(starter, len(self._first))
        return out


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
    #: team -> man -> his starts the season before (`CARRY`).
    previous: dict[str, dict[str, int]] = {}


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


def walk(
    played: Sequence[tuple[Game, GameQb]],
    league: str = "ncaafb",
    last_season: Mapping[str, int] | None = None,
) -> StarterWalk:
    """The availability index's walk over a team's games with passing, in
    order: who its starter is after the last, and how long he's been out.
    `last_season` is the team's starts by man the season before."""
    found = StarterWalk(
        HANDOVER[league], league in KEEP_UNSEEN, last_season, CARRY.get(league)
    )
    for _, qb in played:
        found.step(qb.starter, qb.snaps)
    return found


def doubt(
    team: str,
    played: Sequence[tuple[Game, GameQb]],
    league: str = "ncaafb",
    last_season: Mapping[str, int] | None = None,
) -> Doubt | None:
    """Whether the team's starter, by the availability index's rule
    (`walk`), is in doubt after the last of `played` (its games with
    passing, in order).

    The doubt lasts as long as the index keeps him the starter through his
    absence: until he plays, is seen healthy, or the cover out-starts him.
    """
    if not played:
        return None
    names: dict[str, str] = {}
    for _, qb in played:
        names.update(qb.names)
    walked = walk(played, league, last_season)
    game, last = played[-1]
    est = walked.going_in(last.starter)
    if est is None:
        return None
    count = walked.starts(est)
    who = f"{names.get(est, est)} ({count} start{'s' * (count != 1)})"
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


def _played(
    games: Iterable[Game], starts: Mapping[str, Mapping[str, GameQb]]
) -> dict[str, list[tuple[Game, GameQb]]]:
    """Each team's finished games with passing, in order, with their quarterbacks."""
    played: dict[str, list[tuple[Game, GameQb]]] = {}
    for game in sorted((g for g in games if g.completed), key=lambda g: g.date):
        for team, qb in starts.get(game.game_id, {}).items():
            played.setdefault(team, []).append((game, qb))
    return played


def doubts(
    games: Iterable[Game],
    starts: Mapping[str, Mapping[str, GameQb]],
    league: str = "ncaafb",
    previous: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Doubt]:
    """Every team in doubt going into its next game, by name.

    `games` is the season's schedule; `starts` the per-game quarterbacks,
    keyed like `QbStartsFile.games`. A game without them (no plays, or no
    passing) is passed over, as the availability index passes it.
    """
    found = {}
    for team, history in _played(games, starts).items():
        if (d := doubt(team, history, league, (previous or {}).get(team))) is not None:
            found[team] = d
    return found


def availability(
    team: str,
    played: Sequence[tuple[Game, GameQb]],
    league: str = "ncaafb",
    last_season: Mapping[str, int] | None = None,
) -> float:
    """What the team's next game weighs in the availability index before
    anybody has played it: the chance its starter sits, by how his last game
    went, times what the flag would weigh if he does -- less for each game
    of an absence already behind him (`QB_OUT_DECAY`). 0 with no doubt."""
    found = doubt(team, played, league, last_season)
    if found is None:
        return 0.0
    absent = walk(played, league, last_season).absent
    return AVAILABILITY_OUT[found.kind] * QB_OUT_DECAY**absent


def availabilities(
    games: Iterable[Game],
    starts: Mapping[str, Mapping[str, GameQb]],
    upcoming: Iterable[Game],
    league: str = "ncaafb",
    previous: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, dict[str, float]]:
    """Each team's next game in `upcoming`, weighed by `availability`, by
    game id and team; teams with no doubt are left out, as an index leaves
    out a game nobody missed.

    Only the next game: what a team's last game says about the one after
    that is the next game's to say. Empty outside `PRICED_LEAGUES`.
    """
    if league not in PRICED_LEAGUES:
        return {}
    played = _played(games, starts)
    found: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for game in sorted(upcoming, key=lambda g: g.date):
        for team in (game.home, game.away):
            if team in seen:
                continue
            seen.add(team)
            weight = availability(
                team, played.get(team, []), league, (previous or {}).get(team)
            )
            if weight:
                found.setdefault(game.game_id, {})[team] = weight
    return found

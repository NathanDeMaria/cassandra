"""Which teams played a game without the quarterback who started the last one.

The index is `{league}_qb_out.json`: game id -> the canonical team names whose
expected starter took no snap and was covered for by someone junior to him.
`cassandra.qb_out_build` writes it by walking play-by-play and spells out the
rule; everything on the replay path reads it from here, so
`cassandra.predictor` stays installable without pyarrow -- the same split
`game_control` makes for the same reason.

## The lookahead, said out loud

Whether a quarterback played is read from the plays of the game itself. That
is information nobody has at kickoff, so a model using it is peeking, and
every metric produced with it is optimistic against what a live prediction
could do.

It is deliberate, and the argument for it is that the thing being stood in
for *is* knowable in advance: a quarterback who takes no snap all afternoon
is almost always ruled out beforehand -- an injury, a suspension, a benching
announced in the week. What is missing is the feed that says so, not the
fact. Measuring with the lookahead answers "is this worth wiring an injury
report to?", which is the question that decides whether anyone pays for one.

Read a backtest number from it accordingly. It is an upper bound.

## Live predictions

A game the index has never heard of has nobody out, so a prediction for a
fixture assumes both quarterbacks are fine. That is the right default -- most
of the time they are -- and it means the feature silently does nothing on the
live path unless somebody says otherwise.

Somebody saying otherwise is the point of `QbOutIndex` being constructible
in memory:

    index = QbOutIndex({"401752708": frozenset({"LSU Tigers"})})
    predictor = GlickoPredictor(league, qb_out_penalty=40, qb_out=index)

which is what a caller with an injury report -- or a person who read one --
passes in.
"""

import json
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME

from .types import Matchup

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"

#: The leagues there can be an index for at all: the ones with play-by-play
#: in the bucket. The same tuple `CONTROL_LEAGUES` is, and separate from it
#: on purpose -- these two indexes are built by different sweeps and a league
#: could gain one without the other.
QB_LEAGUES = ("nfl", "ncaafb")


def qb_out_path(league: str) -> Path:
    """Where a league's index lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_qb_out.json"


class QbOutFile(BaseModel):
    """The `{league}_qb_out.json` artifact.

    `games` maps a game id to the teams that were missing their expected
    starter. A game with nobody out is absent rather than present with an
    empty list, which is most of them.
    """

    league: str
    games: dict[str, list[str]]


def read_qb_out_file(league: str) -> QbOutFile | None:
    """A league's artifact as written, or None if there isn't one."""
    path = qb_out_path(league)
    if not path.exists():
        return None
    return QbOutFile.model_validate_json(path.read_text())


@cache
def load_qb_out(league: str) -> Mapping[str, frozenset[str]]:
    """A league's saved availability, empty if it has none.

    Empty is an ordinary answer: four of the six leagues aren't football, and
    a football league whose sweep hasn't run has nothing either. An empty
    index leaves every matchup exactly as it was.

    Cached for the reason `load_anchors` and `load_game_control` are: an
    optimization run builds a predictor per probe and they would all read the
    same file.
    """
    stored = read_qb_out_file(league)
    if stored is None:
        return {}
    return {game: frozenset(teams) for game, teams in stored.games.items()}


def validated_qb_out_penalty(qb_out_penalty: float) -> float:
    """Check a `qb_out_penalty` on its way into a predictor.

    Non-negative, and the sign convention is that this is a *penalty*: it is
    subtracted from the side missing its quarterback. Negative would mean a
    team improves by losing him, which is not a hypothesis worth an hour of
    search.
    """
    if qb_out_penalty < 0:
        raise ValueError(f"qb_out_penalty must be non-negative, got {qb_out_penalty}")
    return qb_out_penalty


class QbOutIndex:
    """Who was missing their quarterback, by game.

    Constructible from a mapping so a caller with an injury report can build
    one for a fixture -- see the module docstring -- and loadable per league
    for a replay.
    """

    def __init__(self, games: Mapping[str, Iterable[str]] | None = None) -> None:
        self._games: dict[str, frozenset[str]] = {
            game: frozenset(teams) for game, teams in (games or {}).items()
        }

    @classmethod
    def for_league(cls, league: str) -> Self:
        """The league's saved index, or an empty one if the sweep hasn't run."""
        return cls(load_qb_out(league))

    def __len__(self) -> int:
        return len(self._games)

    def is_out(self, game_id: str, team: str) -> bool:
        """Whether `team` was missing its expected starter in this game.

        False for a game the index has never seen, which is every fixture --
        a prediction about a game nobody has played assumes both
        quarterbacks are fine.
        """
        return team in self._games.get(game_id, frozenset())

    def differential(self, matchup: Matchup) -> float:
        """+1 if only the away side is missing its quarterback, -1 if only the home.

        Signed toward the home team, like every other matchup adjustment
        here, and 0 when both are out or neither is -- two missing
        quarterbacks is not an edge for anybody.
        """
        home_out = self.is_out(matchup.game_id, matchup.home)
        away_out = self.is_out(matchup.game_id, matchup.away)
        return float(away_out) - float(home_out)

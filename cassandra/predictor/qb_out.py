"""Which teams played a game without their starting quarterback.

The index is `{league}_qb_out.json`: game id -> the canonical team names whose
starter took no snap, and which game of his absence each was.
`cassandra.qb_out_build` writes it by walking play-by-play, and spells out
who counts as the starter; everything on the replay path reads it from here,
so `cassandra.predictor` stays installable without pyarrow -- the same split
`game_control` makes for the same reason.

A flag weighs `QB_OUT_DECAY` to the power of the games of the absence before
it: the whole penalty the first game a backup starts, less each game after,
as the rating takes in the games he plays.

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
fixture assumes both quarterbacks are fine unless somebody says otherwise.
That is the point of `QbOutIndex` being constructible in memory:

    index = QbOutIndex({"401752708": frozenset({"LSU Tigers"})})
    predictor = GlickoPredictor(league, qb_out_penalty=40, qb_out=index)

which is what a caller with an injury report -- or a person who read one --
passes in. Short of news, a fixture can carry a weight instead of a flag: the
chance its starter sits, read off how his last game went
(`cassandra.qb_starts.availability`), times what the flag would weigh. A
starter missing his team's last game with no snap since is out of the next
one most of the time, so `picks.py` prices him that way rather than as fine.
"""

from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
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

#: What each further game of one absence weighs against the game before it.
#: Replayed on ncaafb 2014-2026 at the published fit, carrying an absence at
#: 0.8 against the whole weight every game, brier x1e-5 against flagging its
#: first game only: -4.9 +/- 2.9 with every flag known; -8.2 +/- 2.5 (FBS
#: v FBS -17.1 +/- 8.8) as a live run prices, with `qb_starts.availability`.
#: 1.0 was no better and 0.6 a little worse.
QB_OUT_DECAY = 0.8


def qb_out_path(league: str) -> Path:
    """Where a league's index lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_qb_out.json"


class QbOutFile(BaseModel):
    """The `{league}_qb_out.json` artifact.

    `games` maps a game id to the teams that were missing their expected
    starter. A game with nobody out is absent rather than present with an
    empty list, which is most of them. `runs` says which game of his absence
    each was, 1 for the first; a flag it leaves out is a first game, which is
    how an index built before it reads.

    `fixtures` is each team's next game as the sweep left it, weighed by the
    chance its starter sits (`cassandra.qb_starts.availabilities`): what a
    consumer predicting an unplayed game prices it at. Stale the moment
    another game is played, so a run reading tonight's plays makes its own.
    """

    league: str
    games: dict[str, list[str]]
    runs: dict[str, dict[str, int]] = {}
    fixtures: dict[str, dict[str, float]] = {}


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


@cache
def load_qb_out_runs(league: str) -> Mapping[str, Mapping[str, int]]:
    """Which game of an absence each saved flag was; empty without an index."""
    stored = read_qb_out_file(league)
    return {} if stored is None else stored.runs


@cache
def load_qb_out_fixtures(league: str) -> Mapping[str, Mapping[str, float]]:
    """The saved index's fixtures, weighed; empty without an index."""
    stored = read_qb_out_file(league)
    return {} if stored is None else stored.fixtures


def run_weights(
    runs: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, float]]:
    """What each flag weighs, by game and team: `QB_OUT_DECAY` per earlier
    game of the absence."""
    return {
        game: {team: QB_OUT_DECAY ** (run - 1) for team, run in teams.items()}
        for game, teams in runs.items()
    }


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

    `weights` says what a team counts for in a game, by game and team. A
    flagged team it leaves out counts whole; an unflagged team in it is a
    fixture's chance of its starter sitting, priced but not out.
    """

    def __init__(
        self,
        games: Mapping[str, Iterable[str]] | None = None,
        weights: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self._games: dict[str, frozenset[str]] = {
            game: frozenset(teams) for game, teams in (games or {}).items()
        }
        self._weights: dict[str, dict[str, float]] = {
            game: dict(teams) for game, teams in (weights or {}).items()
        }

    @classmethod
    def for_league(cls, league: str) -> Self:
        """The league's saved index, or an empty one if the sweep hasn't run."""
        return cls(
            load_qb_out(league), run_weights(load_qb_out_runs(league))
        ).with_weights(load_qb_out_fixtures(league))

    @classmethod
    def from_file(cls, stored: QbOutFile) -> Self:
        """An index as a `QbOutFile` reads -- the release's `qb_out.json`, for
        a consumer with no `~/.cassandra`: its flags, their runs, and its
        fixtures."""
        return cls(stored.games, run_weights(stored.runs)).with_weights(stored.fixtures)

    def with_weights(self, weights: Mapping[str, Mapping[str, float]]) -> Self:
        """This index with `weights` laid over its own, game by game and team
        by team: a fixture's chances, say, on the saved flags."""
        merged = {game: dict(teams) for game, teams in self._weights.items()}
        for game, teams in weights.items():
            merged.setdefault(game, {}).update(teams)
        return type(self)(self._games, merged)

    def __len__(self) -> int:
        return len(self._games)

    def is_out(self, game_id: str, team: str) -> bool:
        """Whether `team` was missing its expected starter in this game.

        False for a game the index has never seen, which is every fixture --
        a prediction about a game nobody has played assumes both
        quarterbacks are fine.
        """
        return team in self._games.get(game_id, frozenset())

    def weight(self, game_id: str, team: str) -> float:
        """What `team` counts for in this game: its weight if it has one,
        else 1 if it's flagged out and 0 if not."""
        found = self._weights.get(game_id, {}).get(team)
        if found is not None:
            return found
        return 1.0 if self.is_out(game_id, team) else 0.0

    @contextmanager
    def assuming(self, game_id: str, team: str) -> Generator[None]:
        """`team` missing its starter in `game_id` for the length of the block.

        What a fixture prices at if the news says he sits, asked of a
        predictor that has already replayed: a prediction reads the index
        when it's made, so the answer needs no second replay. He counts whole
        in the block, whatever chance the fixture carried.
        """
        before = self._games.get(game_id)
        weighed = self._weights.get(game_id, {}).get(team)
        self._games[game_id] = (before or frozenset()) | {team}
        self._weights.setdefault(game_id, {})[team] = 1.0
        try:
            yield
        finally:
            if before is None:
                del self._games[game_id]
            else:
                self._games[game_id] = before
            if weighed is None:
                del self._weights[game_id][team]
                if not self._weights[game_id]:
                    del self._weights[game_id]
            else:
                self._weights[game_id][team] = weighed

    def differential(self, matchup: Matchup) -> float:
        """The away side's weight less the home side's: +1 if only the away
        side is missing its quarterback in its first game without him.

        Signed toward the home team, like every other matchup adjustment
        here, and 0 when both are out alike -- two missing quarterbacks is
        not an edge for anybody.
        """
        return self.weight(matchup.game_id, matchup.away) - self.weight(
            matchup.game_id, matchup.home
        )

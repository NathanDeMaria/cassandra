from datetime import datetime
from typing import NamedTuple, Protocol

from endgame.types import Game


class Matchup(Protocol):
    """The part of a game that's known *before* it's played.

    `Game` satisfies this structurally, so callers can pass one straight
    through, but a `predict_game` implementation that takes a `Matchup`
    can't reach the scores.
    """

    @property
    def home(self) -> str: ...

    @property
    def away(self) -> str: ...

    @property
    def neutral_site(self) -> bool: ...

    @property
    def date(self) -> datetime: ...

    @property
    def game_id(self) -> str: ...


class Rating(NamedTuple):
    """One team's standing, in the shape every predictor can express.

    The rating systems don't agree on what a rating *is* -- Elo keeps a single
    number, Glicko keeps a number and its deviation -- so this is the common
    denominator they normalize to on the way out to a release and denormalize
    from on the way back. `rd` is None for the Elo family rather than faked as
    0, because 0 is a meaningful (and very wrong) rating deviation.
    """

    rating: float
    rd: float | None = None


class GameControl(NamedTuple):
    """One game's control: the home team's time-weighted share of winning it.

    The luck-adjusted share, since that is what the sweep writes -- the game
    with its fifty-fifty balls split rather than the one the bounces decided.
    Unnamed in the field, because which reading an index holds is a property
    of the whole file and lives in its `ControlFit` header, not repeated on
    sixteen thousand rows.

    Only the home side, because the away side is `1 - home` by construction
    and carrying both in an artifact is how a `1 -` in the wrong place gets
    shipped. `seconds` is how much regulation clock the average covers, which
    is the part that says whether to trust it: a game whose play-by-play
    stops at halftime reports half a game's worth of seconds.
    """

    home: float
    seconds: int


class Prediction(NamedTuple):
    team1_win_prob: float


class GameResult(NamedTuple):
    prediction: Prediction
    game: Game
    year: int
    week_number: int

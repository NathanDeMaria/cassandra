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


class GameEpa(NamedTuple):
    """One game's EPA per play, one number per offense.

    Expected points added per snap: what each offense moved the ball's value
    by, averaged over the snaps it ran. Unlike `GameControl` these are not
    shares of anything and do not sum to 1 -- they are points, on the
    scoreboard's own scale, and both sides can be positive in a game where
    everybody moved the ball. `home` is what the home offense averaged and
    `away` what the away offense did, each signed for the team with the ball,
    so a home defense that kept forcing punts shows up as a low `away`.

    The *unweighted* reading, since that is what the sweep writes -- every
    snap counted once, garbage time included. `lucky_ones` reports it
    alongside a competitiveness-weighted one and is explicit about which is
    for which job: weighted is the better description of a single game, and
    unweighted "is the one to rank on and the one to add up across a season",
    because the weighting removes noise a season averages away anyway at a
    price in sample that doesn't come back. A rating model is the second job.
    Unnamed in the fields for the reason `GameControl` doesn't name its
    reading either: it is a property of the whole file and lives in the
    `EpaFit` header, not repeated on sixteen thousand rows.

    Both sides, unlike `GameControl` -- there is no `1 -` to recover the away
    number from the home one, and the two denominators genuinely differ.
    `home_plays` and `away_plays` are those denominators, and they are the
    part that says whether to trust the numbers: a per-play average over
    twelve snaps of a weather-shortened game is not the same measurement as
    one over seventy. They are also what turns these averages back into the
    points they came from, which is what a margin-native model wants.
    """

    home: float
    away: float
    home_plays: int
    away_plays: int


class Prediction(NamedTuple):
    team1_win_prob: float


class GameResult(NamedTuple):
    prediction: Prediction
    game: Game
    year: int
    week_number: int

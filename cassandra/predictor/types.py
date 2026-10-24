from datetime import datetime
from typing import NamedTuple, Protocol

from endgame.types import Game


class Matchup(Protocol):
    """The part of a game that's known *before* it's played.

    `Game` satisfies this structurally, so callers can pass one straight
    through, but a `predict_distribution` implementation that takes a `Matchup`
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


class Unit(NamedTuple):
    """One side of a team -- its offense or its defense -- as a rating.

    Always with a deviation, unlike `Rating`: the only model that rates sides
    is a Glicko, and a side's deviation is what says how much it has been
    seen, which the model reads at prediction. See
    `CompoundGlickoPredictor.unit_information`.
    """

    rating: float
    rd: float


class Units(NamedTuple):
    """A team's two sides, on the same scale as its rating."""

    offense: Unit
    defense: Unit


class Rate(NamedTuple):
    """One side's success rate -- a share of its downs -- and its deviation."""

    rate: float
    sd: float


class KindRates(NamedTuple):
    """One kind of down's success rates: the offense's, and what its defense allows."""

    offense: Rate
    defense: Rate


class SuccessRates(NamedTuple):
    """A team's success rate on offense, and the rate its defense allows.

    Each against an average opponent on a neutral field, for a reader: see
    `UnitMarginGlickoPredictor`'s "What a release carries". `dropbacks` and
    `runs` are the same two rates over each kind of down alone, None where
    the games read carried no split.
    """

    offense: Rate
    defense: Rate
    dropbacks: KindRates | None = None
    runs: KindRates | None = None


class PerDown(NamedTuple):
    """One side's EPA per down -- per dropback or run -- and its deviation."""

    epa: float
    sd: float


class KindEpa(NamedTuple):
    """One kind of down's EPA per down: the offense's, and what its defense allows."""

    offense: PerDown
    defense: PerDown


class EpaRates(NamedTuple):
    """A team's EPA per down on offense, and what its defense allows.

    The success rates' twin (`SuccessRates`): each against an average
    opponent on a neutral field, over every down, and over each kind apart.
    """

    offense: PerDown
    defense: PerDown
    dropbacks: KindEpa | None = None
    runs: KindEpa | None = None


class Rating(NamedTuple):
    """One team's standing, in the shape every predictor can express.

    The rating systems don't agree on what a rating *is* -- Elo keeps a single
    number, Glicko keeps a number and its deviation -- so this is the common
    denominator they normalize to on the way out to a release and denormalize
    from on the way back. `rd` is None for the Elo family rather than faked as
    0, because 0 is a meaningful (and very wrong) rating deviation.

    `units` is the offense and defense a compound model keeps under the
    rating, and None for every model that doesn't -- which is all but one,
    so it is last and defaulted and a `Rating(r, rd)` written before it
    existed is the same tuple it always was. Here rather than in a second
    property because this is the seam a release goes through, and a side
    rating that didn't go through it would be lost on the way back.

    `rating` is the number the model steps, which is what `from_ratings`
    rebuilds from. For most models it is also the number the model predicts
    from; `blended` is set by the ones for which it isn't -- the models that
    predict from `rating` and `units` combined -- and `overall` is whichever
    the model predicts from, which is the number to rank on. Derived, so
    `from_ratings` ignores it and a `Rating` built without it is the same
    model.

    `success` and `epa` are for a reader too, and ignored the same way: the
    success rates and EPA per down a units model keeps beside its sides and
    never predicts from.
    """

    rating: float
    rd: float | None = None
    units: Units | None = None
    blended: float | None = None
    success: SuccessRates | None = None
    epa: EpaRates | None = None

    @property
    def overall(self) -> float:
        """The rating the model predicts from: `blended` where there is one."""
        return self.rating if self.blended is None else self.blended


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
    """One game's EPA per play, one number per offense -- two readings of it.

    Expected points added per snap: what each offense moved the ball's value
    by, averaged over the snaps it ran. Unlike `GameControl` these are not
    shares of anything and do not sum to 1 -- they are points, on the
    scoreboard's own scale, and both sides can be positive in a game where
    everybody moved the ball. `home` is what the home offense averaged and
    `away` what the away offense did, each signed for the team with the ball,
    so a home defense that kept forcing punts shows up as a low `away`.

    `home` and `away` are the *unweighted* reading: every snap counted once,
    garbage time included. `lucky_ones` reports it alongside a
    competitiveness-weighted one and is explicit about which is for which
    job: weighted is the better description of a single game, and unweighted
    "is the one to rank on and the one to add up across a season", because
    the weighting removes noise a season averages away anyway at a price in
    sample that doesn't come back. `home_plays` and `away_plays` are the
    denominators, and the part that says whether to trust the numbers: a
    per-play average over twelve snaps of a weather-shortened game is not
    the same measurement as one over seventy. They are also what turns these
    averages back into the points they came from, which is what
    `EpaIndex.margin` does for the margin-native models.

    `home_weighted` and `away_weighted` are the other reading: the same
    bounded EPA over the same snaps, each snap weighted by how much the game
    was still in doubt when it happened -- garbage time adjusted, in the
    phrase. `home_weight` and `away_weight` are the effective samples behind
    them, the sum of the play weights, and the honest part of those numbers
    the way the play counts are for the flat ones: 12 of 70 is an average
    over the first quarter and a half of a rout. The weighted pair can be
    None where the flat one can't -- a team whose every snap came with the
    game already decided has nothing to average -- and defaults to None so a
    `GameEpa` built without it reads as a game with no weighted reading.

    Both readings, because two models want two different things of a game.
    A model blending EPA *into the scoreboard's own target* wants the flat
    one, for `lucky_ones`' reason. A model rating an offense on how it
    played wants the game as it was contested, not as it was run out; see
    `cassandra.predictor.compound`. Neither reading is named in the fields:
    which is which is a property of the whole file, and lives in the
    `EpaFit` header rather than being repeated on sixteen thousand rows.

    `home_success` and `away_success` are a third reading off the same
    snaps: the share of an offense's dropbacks and runs that added expected
    points -- success rate, with `home_success_plays` and
    `away_success_plays` the dropbacks and runs it is over (punts, field
    goals, penalties that wiped the play out, kneels and spikes left out;
    `cassandra.epa_build` says what counts). A sign rather than a size, so
    the one long touchdown that owns a game's EPA is one success among
    sixty, and it predicts the next game better than EPA does for exactly
    that reason; see `cassandra.predictor.unit_glicko`. None, like the
    weighted pair, for a side with no dropbacks or runs, and for a row
    built without it.

    `home_success_weighted` and `away_success_weighted` are the same share
    with each snap weighted as the weighted EPA pair weights it, so a
    blowout's fourth quarter of backups counts for what it is worth. Over
    the same dropbacks and runs; None where the weights sum to nothing,
    like the weighted EPA.

    The `_dropback_` and `_run_` fields split the flat success rate by the
    kind of down: each side's rate over its dropbacks (passes, sacks,
    scrambles) and over its runs, and how many of each. They add back up to
    the flat rate. The `_epa` ones are the same downs' EPA, each play
    bounded as the flat EPA's are -- per down, so over dropbacks and runs
    only, where `home` and `away` are over every snap. For a reader -- no
    model rates on them; see `cassandra.predictor.unit_glicko`'s "What a
    release carries". None for a side with none of that kind, and for a row
    built without them.
    """

    home: float
    away: float
    home_plays: int
    away_plays: int
    home_weighted: float | None = None
    away_weighted: float | None = None
    home_weight: float = 0.0
    away_weight: float = 0.0
    home_success: float | None = None
    away_success: float | None = None
    home_success_plays: int = 0
    away_success_plays: int = 0
    home_success_weighted: float | None = None
    away_success_weighted: float | None = None
    home_dropback_success: float | None = None
    away_dropback_success: float | None = None
    home_dropback_plays: int = 0
    away_dropback_plays: int = 0
    home_run_success: float | None = None
    away_run_success: float | None = None
    home_run_plays: int = 0
    away_run_plays: int = 0
    home_dropback_epa: float | None = None
    away_dropback_epa: float | None = None
    home_run_epa: float | None = None
    away_run_epa: float | None = None


class Prediction(NamedTuple):
    team1_win_prob: float


class GameResult(NamedTuple):
    prediction: Prediction
    game: Game
    year: int
    week_number: int

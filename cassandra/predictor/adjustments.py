"""The three matchup terms, priced together.

A rating says how good a team is. These say what today's game does to the
comparison before the ratings are read: who is better rested, who flew
further, and who is missing a quarterback. All three are properties of the
fixture rather than of either team, all three are knowable before kickoff
(with the caveat `qb_out` spells out about how the index is built), and all
three are added to the home side in rating points.

They are in the update as well as the prediction. A rating model learns
from the gap between what happened and what it expected, and the matchup
terms are part of what it expected: a team that lost without its
quarterback was supposed to, and a team that won at a side coming off a
bye did something harder than the bare ratings say. Elo gets that for free
by updating off the prediction it just made; Glicko recomputes the
expected score inside `glicko_step` and has to be handed the same edge.

One bundle rather than three parameters threaded separately, because the
predictors that want them want all of them and `predict_game` should gain
one line, not three. It follows `PlayBlend` in `blend.py`: a value object
the predictor holds, with the validation and the arithmetic in it rather
than spread across three constructors.

Each weight defaults to 0, which is the model that shipped before any of
this existed. A search prices them and 0 turns any of them off.

## What a consumer of a release gets

The weights round-trip in `state_dict`, so a release carries them. The
*state* behind them does not, and that asymmetry is worth knowing:

- **rest** needs when each team last played, which a release does not carry.
  A predictor rebuilt from one starts with an empty ledger and prices every
  matchup at zero rest until it has walked some games.
- **travel** needs only the two team names and the venue table that ships
  in the package, so it works immediately.
- **qb** needs an index keyed by game id. A fixture is not in the stored one,
  so a live prediction assumes both quarterbacks are fine unless a caller
  passes an index that says otherwise.

Two of the three therefore do nothing on a cold live prediction. That is the
safe direction -- a missing input reads as "no adjustment" rather than as a
wrong number -- and it is the gap to close if any of them earns its place.
"""

from endgame.types import Game

from cassandra.travel import distance_km

from .qb_out import QbOutIndex, validated_qb_out_penalty
from .rest import DEFAULT_REST_ADVANTAGE, RestLedger
from .types import Matchup

#: Rating points per 1,000 km the away side travelled. 0 is off.
#:
#: Per 1,000 rather than per km so the parameter is a number a person can
#: hold: FBS road trips run a median of about 690km and reach 8,000, so a
#: weight of 5 means a cross-country trip is worth 40 rating points and a
#: bus ride to a rival is worth 2.
DEFAULT_TRAVEL_ADVANTAGE = 0.0

#: Rating points taken off a side missing its expected starting quarterback.
DEFAULT_QB_OUT_PENALTY = 0.0

_KM_PER_UNIT = 1000.0


def validated_travel_advantage(travel_advantage: float) -> float:
    """Check a `travel_advantage` on its way into a predictor.

    Non-negative: the term exists because a visitor who travelled further is
    expected to be worse off, and a negative weight would be the claim that
    a long flight helps. The search can already say "no effect" with 0.
    """
    if travel_advantage < 0:
        raise ValueError(
            f"travel_advantage must be non-negative, got {travel_advantage}"
        )
    return travel_advantage


class MatchupAdjustments:
    """Rest, travel and quarterback availability, as one number of rating points.

    Held by `Predictor`, which keeps an inert instance so every predictor can
    ask for the adjustment without a None to guard against -- the same shape
    `_anchors` and `_season_regression` have.
    """

    def __init__(
        self,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        qb_out: QbOutIndex | None = None,
    ) -> None:
        self.rest = RestLedger(rest_advantage)
        self.travel_advantage = validated_travel_advantage(travel_advantage)
        self.qb_out_penalty = validated_qb_out_penalty(qb_out_penalty)
        # An empty index rather than None: "this game has nobody out" is the
        # right answer for a predictor nobody handed an index to, and it is
        # the same answer the stored index gives for a fixture.
        self.qb_out = qb_out if qb_out is not None else QbOutIndex()

    def travel_points(self, matchup: Matchup) -> float:
        """Rating points for how far the away side came.

        0 at a neutral site: the `Game` says a game was neutral and never
        says where it was, so a bowl in Miami and one in Pasadena are the
        same record and neither team's trip can be measured.

        0 for a team the venue table doesn't cover, which is everything
        below FBS -- `distance_km` returns None there rather than guessing,
        and an unknown distance is not a zero-distance trip.
        """
        if not self.travel_advantage or matchup.neutral_site:
            return 0.0
        # The calendar year of the game, which is the season for all but the
        # January tail of one. A relocation is a January apart at worst and
        # nobody moved mid-postseason.
        km = distance_km(matchup.away, matchup.home, matchup.date.year)
        if km is None:
            return 0.0
        return self.travel_advantage * km / _KM_PER_UNIT

    def qb_points(self, matchup: Matchup) -> float:
        """Rating points for a missing quarterback, signed toward the home side."""
        if not self.qb_out_penalty:
            return 0.0
        return self.qb_out_penalty * self.qb_out.differential(matchup)

    def points(self, matchup: Matchup) -> float:
        """Everything this module knows about the matchup, in rating points.

        Added to the home side. Every term is signed the same way -- positive
        favours the home team -- so they compose by adding and a sign error
        in one shows up as that term alone going the wrong way rather than as
        a plausible-looking whole.
        """
        return (
            self.rest.adjustment(matchup)
            + self.travel_points(matchup)
            + self.qb_points(matchup)
        )

    def record(self, game: Game) -> None:
        """Note a played game. Called from `update_game` after the prediction.

        Only the rest ledger has anything to remember: travel is a lookup and
        the quarterback index is built offline.
        """
        self.rest.record(game)

    def pass_season(self) -> None:
        """Cross a season boundary."""
        self.rest.reset()

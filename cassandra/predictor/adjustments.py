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

## Weights and sources

Two things vary here, for different reasons, so they are held apart. A
**weight** -- `rest_advantage`, `travel_advantage`, `qb_out_penalty` -- is
what a term is worth. It is fit by a search, it is a plain float, and it
round-trips in `state_dict` so a release carries it. A **source** is where a
term's fact comes from for a given game: the league's built quarterback
index, the dates a replay has walked. Sources are bundled in
`MatchupSources`, handed to a predictor at construction, and never go through
a config -- none of them is a number to search over.

That split is what keeps `predict_game(matchup)` the only interface. A
what-if about Saturday's fixture and a replay of 2014 are the same call
against the same predictor; they differ in which sources it was built with. A
caller who knows something a replay cannot derive says so by swapping one in
(`StatedRest`, a `QbOutIndex` built in memory) rather than by reaching for a
second prediction path. And a *new* kind of matchup fact is a field on
`MatchupSources` and a term here -- no predictor's constructor changes.

## What a consumer of a release gets

The weights, and no sources. That asymmetry is worth knowing:

- **rest** needs when each team last played, which a release does not carry.
  A predictor rebuilt from one starts with an empty ledger and prices every
  matchup at zero rest until it has walked some games -- or until somebody
  hands it a `StatedRest`.
- **travel** needs only the two team names and the venue table that ships
  in the package, so it works immediately.
- **qb** needs an index keyed by game id. A fixture is not in the stored one,
  so a live prediction assumes both quarterbacks are fine unless a caller
  passes an index that says otherwise.

Two of the three therefore do nothing on a cold live prediction nobody has
told anything. That is the safe direction -- a missing input reads as "no
adjustment" rather than as a wrong number.
"""

from typing import NamedTuple, Self

from endgame.types import Game

from cassandra.travel import distance_km

from .qb_out import QbOutIndex, validated_qb_out_penalty
from .rest import (
    DEFAULT_REST_ADVANTAGE,
    RestLedger,
    RestSource,
    validated_rest_advantage,
)
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


class MatchupSources(NamedTuple):
    """Where the matchup terms get their facts, as one bundle.

    One parameter on a predictor instead of one per kind of fact, which is the
    point: `qb_out` used to be named in five constructors, and the next fact
    would have been named in five more. A caller adjusts one field and leaves
    the rest alone -- `_replace` is the whole idiom:

        MatchupSources.for_league(league)._replace(
            qb_out=QbOutIndex({game_id: ["LSU Tigers"]})
        )

    A tuple because these are inputs a caller assembles and hands over, not
    state this holds: the mutable thing is the `RestLedger` *inside* it, which
    the replay walks.
    """

    qb_out: QbOutIndex
    rest: RestSource

    @classmethod
    def for_league(cls, league: str) -> Self:
        """What a replay wants: the league's saved index, and a live ledger.

        The ledger is empty rather than absent -- it has nothing to say until
        games have been walked into it, which is exactly what a replay is
        about to do.
        """
        return cls(qb_out=QbOutIndex.for_league(league), rest=RestLedger())

    @classmethod
    def empty(cls) -> Self:
        """Sources that know nothing, so every term prices at 0.

        What a predictor exposing none of the weights gets, for the reason
        `Predictor` holds an inert `MatchupAdjustments`: answers of 0 beat a
        None to guard against at every call site. Built fresh each time rather
        than shared, because a `RestLedger` accumulates and two predictors
        must not walk into the same one.
        """
        return cls(qb_out=QbOutIndex(), rest=RestLedger())


def resolved_sources(league: str, sources: MatchupSources | None) -> MatchupSources:
    """The sources a predictor should use, given what it was handed.

    A free function for the reason `resolved_anchors` is one, and it reads the
    same: `None` means "the league's own", which is what a replay wants, and a
    caller with something to state passes a bundle and gets it back untouched.
    """
    if sources is None:
        return MatchupSources.for_league(league)
    return sources


class MatchupAdjustments:
    """Rest, travel and quarterback availability, as one number of rating points.

    Held by `Predictor`, which keeps an inert instance so every predictor can
    ask for the adjustment without a None to guard against -- the same shape
    `_anchors` and `_season_regression` have.

    The weights are its own; the facts they price come from `sources`. See the
    module docstring for why those are two different things.
    """

    def __init__(
        self,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        sources: MatchupSources | None = None,
    ) -> None:
        self.rest_advantage = validated_rest_advantage(rest_advantage)
        self.travel_advantage = validated_travel_advantage(travel_advantage)
        self.qb_out_penalty = validated_qb_out_penalty(qb_out_penalty)
        # Empty rather than None: "nobody is out and nobody is rested" is the
        # right answer for a predictor nobody told anything, and it is the
        # same answer the stored sources give for a fixture.
        self.sources = sources if sources is not None else MatchupSources.empty()

    def rest_points(self, matchup: Matchup) -> float:
        """Rating points for the rest gap, signed toward the home side.

        Applied at a neutral site as well, unlike a home advantage: nobody is
        at home in a bowl and both teams still arrived on different rest.
        """
        if not self.rest_advantage:
            return 0.0
        return self.rest_advantage * self.sources.rest.rested_side(matchup)

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
        return self.qb_out_penalty * self.sources.qb_out.differential(matchup)

    def points(self, matchup: Matchup) -> float:
        """Everything this module knows about the matchup, in rating points.

        Added to the home side. Every term is signed the same way -- positive
        favours the home team -- so they compose by adding and a sign error
        in one shows up as that term alone going the wrong way rather than as
        a plausible-looking whole.
        """
        return (
            self.rest_points(matchup)
            + self.travel_points(matchup)
            + self.qb_points(matchup)
        )

    def record(self, game: Game) -> None:
        """Note a played game. Called from `update_game` after the prediction.

        Only the rest source has anything to remember: travel is a lookup and
        the quarterback index is built offline. A source that was told its
        answer rather than walking to it does nothing here.
        """
        self.sources.rest.record(game)

    def pass_season(self) -> None:
        """Cross a season boundary."""
        self.sources.rest.reset()

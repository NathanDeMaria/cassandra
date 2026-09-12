"""Glicko on the result, with an offense and a defense hanging off each team.

`GlickoPredictor` rates a team by whether it won. This keeps that rating --
the *parent*, updated exactly as Glicko updates it -- and grows two *children*
under it: an offense and a defense, each a Glicko (rating, deviation) pair of
its own, rated not on the game's result but on EPA per play.

A football game is two contests played at once. The home offense takes the
ball against the away defense, and what it averaged per snap -- `GameEpa.home`
-- is the result of that contest, a number nobody has to infer from the
scoreboard. The away offense against the home defense is the other one, and
`GameEpa.away` is its result. So where the parent sees one contest per game,
the children see two, and each of the two has a side that gets scored *as if
it were a Glicko match*: the offense's EPA per play, squashed through a
logistic to [0, 1], is what the offense scored and one minus it is what the
defense did.

Putting EPA on the parent's scale
---------------------------------

The parent's target for a game is `sigmoid(margin / sigmoid_scale)` and its
expected score is Glicko's logistic of a rating gap, so a rating gap *is* an
expected points margin, at whatever slope the league's fit found. For the
children to be on that scale their contests have to be scored in the same
points through the same squash, and they are:

- An offense's per-play average times its snaps is the points it added over
  the game. Minus what an average offense would have added over the same
  snaps -- `epa_center`, a running mean of every offense seen so far -- it is
  the offense's margin over average, and the two offenses' margins in a game
  difference to the game's EPA margin, which `glicko_blend` measured at 0.86
  of the scoreboard's spread. `epa_scale` is the exchange rate between the
  two kinds of points.

- A contest is half a game. With a team's unit rating defined as the mean of
  its offense and defense, the two contests' gaps -- home offense less away
  defense, away offense less home defense -- sum to *twice* the gap between
  the two teams' unit means, so a contest's points are read at twice the
  team slope: `sigmoid(2 * points / sigmoid_scale)`. Without the 2 every
  unit gap comes out half the parent's for the same strength.

Which makes the anchors priors for both sides. A division fit says where a
D-III *team* enters, and a child on the team's scale can enter there too, so
the first FBS-vs-FCS contest in the index is scored with the gap it deserves
rather than between two units at an uninformative 0. Concretely, the prior a
child sits on is

    parent_share * parent + (1 - parent_share) * anchor

held as an offset from that prior. At `parent_share` 1 the children are
residuals on top of the record; at 0 they are ratings of their own, seeded
where the parent was seeded. Either way 0 offset is "nothing known" and the
offseason pulls toward it.

Measured, the scale holds: with the units alone (`unit_weight` 0, so they are
learning without speaking) the unit gap tracks the parent's gap at +0.91
correlation with a slope of 0.67 across 20,860 ncaafb games, and across
teams the unit mean correlates +0.87 with the parent rating.

How the children speak
----------------------

Two ratings on one scale blend, so at prediction a team is rated

    (1 - w) * parent + w * (offense + defense) / 2,   w = unit_weight * confidence

and the parent's own arithmetic -- home edge, matchup adjustments, Glicko's
logistic of the gap -- runs on that. `confidence` is one minus the units'
share of the variance they started with: 0 for a pair still at the initial
deviation, toward 1 as it tightens. It is what keeps a team the index has
never had a play for at its record, which is most of an ncaafb schedule and
all of it before 2006; without it the blend regresses every such team toward
its anchor by `unit_weight`, and costs more on the games it can't see than it
earns on the ones it can (measured: +0.00019 overall against -0.00012 on
the EPA games, at `unit_weight` 0.1).

At `unit_weight` 0 this is `GlickoPredictor` exactly, game by game -- the
same property `BlendedGlickoPredictor` has at `play_weight` 0 and for the
same reason, so that a search's own zero is the baseline's score on the same
games in the same process. The children never feed back into the parent's
update, and the parent never sees an EPA number.

What this asks that `BlendedGlickoPredictor` doesn't
-----------------------------------------------------

That module blends EPA into the *target* of the same one rating, and measured
it to be a second, slightly noisier copy of the final score: real, and worth
about 0.0001 brier. The reading it ends on is that EPA's net margin says what
the scoreboard says. This model doesn't use the net margin. It uses the two
sides separately, and the question it can answer that a single rating cannot
is whether a team's *shape* -- great offense, bad defense -- says anything
about its next game that its record doesn't. A one-dimensional rating has to
call two teams with the same record the same; this one doesn't have to.

Like the other play-by-play models, nothing here builds a game that wasn't
played: the replay records the result, and EPA enters where the ratings are
decided and nowhere near where the model is scored.

What it was measured to be worth
--------------------------------

Replayed on ncaafb 2002-2023 off the local season cache (2021, 2024 and 2025
skipped for mis-grouped weeks), holding `glicko_full`'s fitted parameters and
the same saved opponent priors, so the parent is the baseline exactly and
`unit_weight` 0 reproduces its 0.158621. Every number is against that, and
the EPA column is over the 20,860 games the index has.

    parent_share  epa_scale  unit_weight   d brier   d brier, EPA games
             0.0        2.5         0.10  -0.000063         -0.000171
             0.0        1.6         0.10  -0.000060         -0.000159
             0.0        4.0         0.10  -0.000061         -0.000167
             0.0        1.0         0.10  -0.000048         -0.000119
             0.0        1.0         0.25  +0.000027         +0.000008
             0.0        1.0         0.50  +0.000590         +0.001139
             0.0        1.0         1.00  +0.003652         +0.007555
             0.5        1.0         0.10  -0.000042         -0.000098
             1.0        1.0         0.10  -0.000017         -0.000034
             1.0        1.0         1.00  +0.004674         +0.011031

Three readings. **The weight wants to be small**: a tenth of the rating,
with a quarter already break-even and the units alone (`unit_weight` 1) a
model 0.004 worse than the parent even though they are on its scale and
correlate with it at 0.87. That is the finding of `glicko_blend` arriving by
a different road -- EPA is a noisier measurement of the thing the scoreboard
measures, worth the variance reduction of averaging in a little of it and
nothing beyond. **The anchor is the better prior**: `parent_share` 0 beats
1 at every weight, and 1 at weight 1 is the worst row in the table. Scoring
a child against its team's record makes it learn the part of the record the
plays don't show, and that part predicts less well than the plays do on
their own. **`epa_scale` is flat above 1**: 1.6 through 4.0 are within
0.000003 of each other. Past ~2 the squash is saturated and a contest is
scored on who won it, which is `binary_score`'s answer, and it is a little
better than the points. `unit_initial_rd` above the parent's `initial_rd`
costs on the non-EPA games -- a looser prior takes longer to earn
confidence and moves further when it does -- so it defaults to the parent's
and stays out of the search.

The gain is about half `BlendedGlickoPredictor`'s -0.00013 and, like it,
about a hundredth of the 0.009 that separates ncaafb's two best models.

Where this came from is worth a paragraph because it is the lesson. The
first version scored a contest as `sigmoid(epa_per_play / 0.2)` -- a
logistic of a per-play average over an arbitrary scale -- and added the
children to the parent's gap at a free exchange rate. Scored against the
parent (`parent_share` 1) it lost at every setting from a whisper to full
weight, with a team's offense and defense correlating at +0.84 and both
falling with the number of EPA games the team had played: two children
moving together, by more the more a team was seen, is a per-team correction
and what it was correcting was the parent. A 400-point favorite was expected
to post an EPA score no offense posts on average, so both its units were
marked down every game it was favored. Scored against nothing (children as
ratings in their own arbitrary currency) it helped by 0.00004. Putting the
children in the parent's currency, as above, is what let the anchors seed
them and what turned the residual reading from destructive into merely
worse; fading unseen units out of the blend is what let the seeded children
be used at all. Between them they are the improvement from 0.00004 to
0.00006.

**The signal is real.** Permute which game each `GameEpa` belongs to --
both sides kept together, every marginal preserved, only the correspondence
to the game played destroyed -- and replay the best row fifteen times:

    shuffles  null mean   vs off      real is
          15   0.158793  +0.000172   8.6 sd better, 0/15 matched it

Note the sign on the null, as `glicko_blend` notes it: a shuffled index is
*worse than no index at all*, by nearly three times what the real one is
better by. Units seeded on real anchors and confidently rated on the wrong
games are a liability, not noise, so this is a narrow well and not a flat
direction any plausible number would do for. It is also the operational
guarantee that an index which ever went misaligned would show up as
degradation rather than as nothing.

What is not established: the optimum, which this scan bracketed rather than
found, and whether nfl behaves the same way; the configs in `models/` are
what will say.

What a release carries
----------------------

`ratings` hands each team's `Rating` out with its two sides attached, as
absolutes on the team's scale, and `from_ratings` puts them back -- so a
release rehydrates to a model that predicts as the publisher's did. A team
the index never had a play for goes out with no sides and comes back with
none, which is confidence 0 and the parent alone. The one thing a release
does not carry is the running center, which only matters to a rebuilt model
that keeps updating; see `from_ratings`.
"""

import math
from collections.abc import Mapping
from typing import Any, NamedTuple, Self

from endgame.types import Game

from ..scoring import DEFAULT_SIGMOID_SCALE
from .adjustments import DEFAULT_QB_OUT_PENALTY, DEFAULT_TRAVEL_ADVANTAGE
from .base_predictor import Anchor
from .blend import validated_fraction, validated_scale
from .epa import EpaIndex
from .glicko import GlickoPredictor, _Rating, glicko_step
from .opponent_prior import OpponentPriorManager
from .qb_out import QbOutIndex
from .rest import DEFAULT_REST_ADVANTAGE
from .types import GameEpa, Matchup, Prediction, Rating, Unit, Units

# How much of a fully-known team's rating, at prediction, is its units rather
# than its record: `(1 - w) * parent + w * (offense + defense) / 2`, with `w`
# this times the units' confidence.
#
# A fraction, because the two are on one scale and a blend of two opinions is
# what combining them means. A tenth rather than the visible-blend half
# `DEFAULT_PLAY_WEIGHT` reasons its way to, because this one is measured: a
# quarter is already break-even and a half costs 0.0006 -- see the module
# docstring -- so a hand-built one at a half would be the model the numbers
# say not to build. Not 0, which would make it silently identical to the
# model it subclasses.
DEFAULT_UNIT_WEIGHT = 0.1

# Points of scoreboard margin per point of EPA.
#
# 1.0 takes EPA at face value as the points it claims to be, which is where
# `margin_blend` starts too and for the same reason: it is the one setting
# with a meaning. It is not where the numbers say to stay -- EPA's implied
# margin has 0.86 of the final margin's spread in both football leagues
# (`glicko_blend` measured it), because every play is bounded at the clip and
# blowouts are understated -- so a search should expect to find something
# above 1, and 1/0.86 = 1.16 is the value that matches the two spreads.
DEFAULT_EPA_SCALE = 1.0

# How much of the parent a child is scored against, as against its anchor:
# the prior a child sits on is `parent_share * parent + (1 - parent_share) *
# anchor`. 1 makes the children residuals on top of the record, 0 makes them
# ratings of their own seeded where the parent was seeded. A fraction rather
# than a switch so a search can walk between the two and say where the record
# stops being worth listening to. Measured, 0 wins at every weight -- the
# module docstring has the table -- so the anchor is the default prior.
DEFAULT_PARENT_SHARE = 0.0


class _Units(NamedTuple):
    """One team's two children, each on the parent's rating scale.

    Held as offsets from the prior they are scored against (see
    `CompoundGlickoPredictor._prior`), so 0 is "nothing known" whatever the
    prior is and the season rollover has one number to pull toward.
    `get_units` hands back the absolute pair.
    """

    offense: _Rating
    defense: _Rating


class CompoundGlickoPredictor(GlickoPredictor):
    """Glicko on results, with an offense and a defense per team rated on EPA.

    The parent's every method does what it did. What this adds is a second
    dictionary of ratings, `_units`, updated by `_update_units` from the EPA
    index after the parent has updated from the score, and read by
    `predict_game` as a second rating for each team to blend with the first.

    `scoring_method` and `sigmoid_scale` are the parent's and are passed
    through. The children read `sigmoid_scale` too -- it is what puts a
    points margin on the rating scale, and they have to share it with the
    parent or they are not on the parent's scale -- whatever `scoring_method`
    is. The football configs pin it to sigmoid, which is the setting under
    which the two are on identical footing.

    `unit_initial_rd` is `None` for "the parent's `initial_rd`": a unit nobody
    has seen play is exactly as unknown as a team nobody has, and the parent's
    fit already says how unknown that is for the league.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        k: float = 65,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        scoring_method: str = "binary",
        sigmoid_scale: float = DEFAULT_SIGMOID_SCALE,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        unit_weight: float = DEFAULT_UNIT_WEIGHT,
        epa_scale: float = DEFAULT_EPA_SCALE,
        parent_share: float = DEFAULT_PARENT_SHARE,
        unit_initial_rd: float | None = None,
        opponent_prior_manager: OpponentPriorManager | None = None,
        qb_out: QbOutIndex | None = None,
        ratings: dict[str, _Rating] | None = None,
        units: dict[str, _Units] | None = None,
        # The running mean the contests are centered on, as (sum, count) of
        # every offense's per-play average seen so far. Carried so a loaded
        # model keeps centering where the replay left it.
        epa_seen: tuple[float, int] = (0.0, 0),
        anchors: Mapping[str, Anchor] | None = None,
        game_epa: EpaIndex | None = None,
    ) -> None:
        super().__init__(
            league,
            home_advantage=home_advantage,
            k=k,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            scoring_method=scoring_method,
            sigmoid_scale=sigmoid_scale,
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            qb_out=qb_out,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            ratings=ratings,
            anchors=anchors,
        )
        self._unit_weight = validated_fraction("unit_weight", unit_weight)
        self._epa_scale = validated_scale("epa_scale", epa_scale)
        self._parent_share = validated_fraction("parent_share", parent_share)
        self._unit_initial_rd = (
            initial_rd
            if unit_initial_rd is None
            else validated_scale("unit_initial_rd", unit_initial_rd)
        )
        # Units start empty rather than seeded from the prior manager the way
        # the parent's ratings are: a unit's prior is the team's anchor or
        # its parent, and 0 on top of either is "nothing known".
        self._units: dict[str, _Units] = units if units is not None else {}
        self._epa_sum, self._epa_count = epa_seen
        # Defaulted rather than required, like `opponent_prior_manager`: every
        # caller wants the league's saved index, and a test that wants none
        # passes `EpaIndex()`. Not carried in `state_dict` -- see below.
        self._game_epa = game_epa or EpaIndex.for_league(league)

    @property
    def unit_weight(self) -> float:
        """How much of a team's rating, at prediction, is its units."""
        return self._unit_weight

    @property
    def parent_share(self) -> float:
        """How much of the parent, as against the anchor, a child sits on."""
        return self._parent_share

    @property
    def epa_center(self) -> float:
        """EPA per play of the average offense seen so far; 0 before any.

        A running mean rather than a constant, so it needs no per-league
        pin and reads nothing from games the replay hasn't reached. It is
        what makes the anchors real priors for both sides: centered here, an
        average offense and an average defense both sit at their team's
        anchor rather than drifting apart from it by a league-wide constant.
        """
        if not self._epa_count:
            return 0.0
        return self._epa_sum / self._epa_count

    def _prior(self, team: str, parent: _Rating | None = None) -> float:
        """The rating a child of `team` is measured from.

        `parent_share` of the team's parent rating and the rest of its
        anchor. Both are on the rating scale, so the child that sits on top
        is too -- which is the whole point, and what lets an anchor fit for
        a division be a prior for that division's offenses and defenses.

        `parent` is passed in by the update, which needs the rating *before*
        the game moved it; everyone else reads the current one.
        """
        if not self._parent_share:
            return self.anchor(team)
        rating = (parent or self.get_rating(team)).rating
        return self._parent_share * rating + (1 - self._parent_share) * self.anchor(
            team
        )

    def _offsets(self, team: str) -> _Units:
        return self._units.get(
            team,
            _Units(
                _Rating(0.0, self._unit_initial_rd),
                _Rating(0.0, self._unit_initial_rd),
            ),
        )

    def get_units(self, team: str) -> _Units:
        """This team's offense and defense, on the parent's rating scale.

        A team nobody has EPA for sits at its prior with the initial
        deviation -- the same answer as a parent rating sitting at its
        anchor, one level down.
        """
        prior = self._prior(team)
        offsets = self._offsets(team)
        return _Units(
            _absolute(prior, offsets.offense), _absolute(prior, offsets.defense)
        )

    def unit_rating(self, team: str) -> float:
        """The team as its units see it: the mean of offense and defense.

        The mean and not the sum because of how the two contests add up. A
        team's game is its offense against their defense plus their offense
        against its defense, and the two gaps together come to twice the gap
        between the two teams' unit means -- so the mean is the number on
        the parent's scale, and `_contest_score` doubles the other side of
        that identity.
        """
        units = self.get_units(team)
        return (units.offense.rating + units.defense.rating) / 2

    def unit_gap(self, matchup: Matchup) -> float:
        """How much better the home side is by its units, in rating points.

        The second opinion, on the parent's scale and before `unit_weight`,
        so a reader can set it beside the parent's gap.
        """
        return self.unit_rating(matchup.home) - self.unit_rating(matchup.away)

    def unit_confidence(self, team: str) -> float:
        """How much this team's units have learned, in [0, 1].

        0 for a pair still at the initial deviation -- nothing seen, or a
        release that never carried them -- and toward 1 as the deviations
        tighten: one minus the units' share of the variance they started
        with. It is what keeps a team the index has never had a play for at
        its parent's rating, rather than blended toward its anchor by a unit
        that is nothing but the anchor. That is most of an ncaafb schedule
        and all of it before 2006, so without this the blend costs more on
        the games it can't see than it earns on the ones it can.

        Glicko already keeps the number this needs. The parent has no
        matching factor because it is the base estimate: a team the parent
        has never seen is at its anchor and the blend of an anchor with an
        anchor is the anchor.
        """
        offsets = self._offsets(team)
        variance = (
            offsets.offense.rating_deviation**2 + offsets.defense.rating_deviation**2
        ) / 2
        return 1 - variance / self._unit_initial_rd**2

    def _blended_rating(self, team: str) -> float:
        """What a team is rated, for a prediction: record and units, blended.

        `unit_weight` is how much of a fully-known unit pair's opinion is
        taken; `unit_confidence` is how fully known this pair is.
        """
        parent = self.get_rating(team).rating
        if not self._unit_weight:
            return parent
        weight = self._unit_weight * self.unit_confidence(team)
        return (1 - weight) * parent + weight * self.unit_rating(team)

    def predict_game(self, matchup: Matchup) -> Prediction:
        """The parent's arithmetic on a blended rating.

        The parent's own prediction is a logistic of a rating gap, and the
        only thing this changes is which rating: a blend of the record's and
        the units', which are on one scale and so can be blended. The home
        edge and the matchup adjustments are the parent's, applied once, and
        the test that `unit_weight` 0 reproduces the parent game by game is
        what guards the repetition.
        """
        home_rating = self._blended_rating(matchup.home)
        if not matchup.neutral_site:
            home_rating += self._home_advantage
        home_rating += self.matchup_adjustment(matchup)
        away_rating = self._blended_rating(matchup.away)
        win_prob = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        return Prediction(team1_win_prob=win_prob)

    def update_game(self, game: Game) -> Prediction:
        """The parent's update, then the children's.

        The parents are read first and handed to `_update_units`, because
        `super().update_game` moves them and a unit's expected score belongs
        against the team its opponent was known to be when the contest was
        played -- the same "predict, then learn" ordering the parent itself
        follows. The children go second so the parent's `_update_rating`
        calls, which read both teams' ratings before writing either, see the
        state they expect.
        """
        home_parent = self.get_rating(game.home)
        away_parent = self.get_rating(game.away)
        prediction = super().update_game(game)
        self._update_units(game, home_parent, away_parent)
        return prediction

    def _contest_score(self, epa_per_play: float, plays: int) -> float:
        """What an offense's game counts as, in [0, 1], on the parent's scale.

        Per-play average times snaps is the points the offense added, and
        minus what an average offense would have added over the same snaps
        it is the offense's margin over average -- the two offenses' margins
        difference to the game's EPA margin, which is what puts them in the
        scoreboard's points. `epa_scale` is the exchange rate between the two
        kinds of points; see `DEFAULT_EPA_SCALE`.

        Then the parent's own squash, `sigmoid(points / sigmoid_scale)`, at
        *twice* the points. Twice because a contest is half a game: the two
        contests' gaps sum to twice the gap between the teams' unit means
        (see `unit_rating`), so a contest's points have to be read at twice
        the team slope for the units to land on the team scale. Without the
        2 every unit gap would be half as large as the parent's for the same
        strength, and the blend would be mixing currencies again.
        """
        points = (epa_per_play - self.epa_center) * plays * self._epa_scale
        return 1 / (1 + math.exp(-2 * points / self._sigmoid_scale))

    def _see(self, epa: GameEpa) -> None:
        """Fold a game's two offenses into the running center.

        Before the game is scored, not after, so the very first contest is
        centered on its own game rather than on 0 -- and after that it is
        one game in a thousand either way. It has already been played, so
        nothing here is read before it happened.
        """
        self._epa_sum += epa.home + epa.away
        self._epa_count += 2

    def _update_units(
        self, game: Game, home_parent: _Rating, away_parent: _Rating
    ) -> None:
        """Run the two unit contests, if the game has EPA to score them by.

        `home_parent` and `away_parent` are the parents *as they stood before
        the game*, passed in rather than re-read because `update_game` has
        already moved them. A unit's expected score is against the team its
        opponent was known to be when the contest was played, not the one it
        became by playing it.

        Each contest goes through `glicko_step` with both sides at prior plus
        offset, so the step sees absolute ratings on the parent's scale, and
        the offset stored is what the step returned minus the prior it was
        built on. The deviation is the child's own: the parent's is not
        folded in, because the parent's uncertainty is the parent's business
        and a child that inherited it would look twice as unsure as the
        evidence says.

        The home edge is the parent's `home_advantage`, whole, on each
        contest -- the parent's own update applies it the same way, and the
        two contests' edges sum to twice one edge exactly as their gaps sum
        to twice the team gap. Like the parent's update, not the matchup
        adjustments, which the parent reserves for the prediction.
        """
        epa = self._game_epa.get(game.game_id)
        if epa is None:
            return
        self._see(epa)
        home_adj = 0 if game.neutral_site else self._home_advantage
        home_prior = self._prior(game.home, home_parent)
        away_prior = self._prior(game.away, away_parent)
        home = self._offsets(game.home)
        away = self._offsets(game.away)

        home_offense = _absolute(home_prior, home.offense)
        home_defense = _absolute(home_prior, home.defense)
        away_offense = _absolute(away_prior, away.offense)
        away_defense = _absolute(away_prior, away.defense)

        home_ball = self._contest_score(epa.home, epa.home_plays)
        away_ball = self._contest_score(epa.away, epa.away_plays)

        self._units[game.home] = _Units(
            offense=_offset(
                home_prior,
                glicko_step(home_offense, away_defense, home_ball, home_adj),
            ),
            defense=_offset(
                home_prior,
                glicko_step(home_defense, away_offense, 1 - away_ball, home_adj),
            ),
        )
        self._units[game.away] = _Units(
            offense=_offset(
                away_prior,
                glicko_step(away_offense, home_defense, away_ball, -home_adj),
            ),
            defense=_offset(
                away_prior,
                glicko_step(away_defense, home_offense, 1 - home_ball, -home_adj),
            ),
        )

    def _grow_unit_rd(self, units: _Units, increase: float) -> _Units:
        """Both children's deviations widened by `increase`, capped like the parent's."""
        return _Units(
            *(
                _Rating(
                    unit.rating,
                    min(
                        self._unit_initial_rd,
                        math.sqrt(unit.rating_deviation**2 + increase**2),
                    ),
                )
                for unit in units
            )
        )

    def pass_week(self) -> None:
        super().pass_week()
        self._units = {
            team: self._grow_unit_rd(units, self._weekly_rd_increase)
            for team, units in self._units.items()
        }

    def _roll_over(self) -> None:
        """The parent's offseason, and the children's.

        A child's offset regresses toward 0 -- toward its prior, which is the
        team's anchor or its parent or a mix -- by the same
        `season_regression` the parent uses toward its anchor. Not through
        `regress`, whose target is the anchor itself: the offset is already
        measured from there.
        """
        super()._roll_over()
        keep = 1 - self._season_regression
        self._units = {
            team: self._grow_unit_rd(
                _Units(
                    _Rating(
                        keep * units.offense.rating, units.offense.rating_deviation
                    ),
                    _Rating(
                        keep * units.defense.rating, units.defense.rating_deviation
                    ),
                ),
                self._season_rd_increase,
            )
            for team, units in self._units.items()
        }

    def state_dict(self) -> dict[str, Any]:
        """The parent's state, the parameters this adds, and the children.

        The children go in because a release that lost them would predict as
        the parent alone -- see the module docstring. They are four floats per
        team against the parent's two, small enough to carry, and they are
        the offsets rather than the absolutes so a loaded model rebuilds the
        same absolutes from the same priors. The EPA index stays out for the
        reason `ControlGlickoPredictor` gives: tens of thousands of floats
        against a release that is otherwise a page of ratings.
        """
        return {
            **super().state_dict(),
            "unit_weight": self._unit_weight,
            "epa_scale": self._epa_scale,
            "parent_share": self._parent_share,
            "unit_initial_rd": self._unit_initial_rd,
            "epa_seen": [self._epa_sum, self._epa_count],
            "units": {
                team: [
                    units.offense.rating,
                    units.offense.rating_deviation,
                    units.defense.rating,
                    units.defense.rating_deviation,
                ]
                for team, units in self._units.items()
            },
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        params = dict(data)
        params["units"] = {
            team: _Units(_Rating(u[0], u[1]), _Rating(u[2], u[3]))
            for team, u in params.pop("units", {}).items()
        }
        seen = params.pop("epa_seen", (0.0, 0))
        params["epa_seen"] = (float(seen[0]), int(seen[1]))
        return super().from_state_dict(params)

    @property
    def ratings(self) -> dict[str, Rating]:
        """The parent's ratings, each carrying its two sides where there are any.

        Absolutes, not offsets: a release reader wants a 1650 offense to
        mean what a 1650 team does, and the prior an offset is measured from
        is this model's business. Only teams the index has had a play for
        carry sides; the rest are None rather than a pair sitting at the
        prior with the initial deviation, which a consumer could not tell
        from a measured pair that happens to sit there.
        """
        return {
            team: rating._replace(
                units=self._absolute_units(team) if team in self._units else None
            )
            for team, rating in super().ratings.items()
        }

    def _absolute_units(self, team: str) -> Units:
        units = self.get_units(team)
        return Units(
            Unit(units.offense.rating, units.offense.rating_deviation),
            Unit(units.defense.rating, units.defense.rating_deviation),
        )

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        """The parent's rebuild, then the sides put back under their teams.

        The release holds absolutes, so each side goes back to an offset from
        the prior *this* predictor computes -- its parent rating and whatever
        anchors it has -- and comes back out of `get_units` as the absolute
        that went in, whether or not the anchors match the publisher's.

        What a release does not carry is the running center, so a rebuilt
        model that goes on to update from new games recenters from those. It
        converges within a few dozen offenses and cancels in every gap the
        model reads, so a prediction never sees it; a rating read off such a
        model could sit a few points from where the publisher's would.
        """
        predictor = super().from_ratings(league, ratings, **params)
        for team, rating in ratings.items():
            if rating.units is None:
                continue
            prior = predictor._prior(team)
            predictor._units[team] = _Units(
                _offset(prior, _Rating(*rating.units.offense)),
                _offset(prior, _Rating(*rating.units.defense)),
            )
        return predictor


def _absolute(prior: float, offset: _Rating) -> _Rating:
    """A child as `glicko_step` should see it: its prior plus its offset."""
    return _Rating(prior + offset.rating, offset.rating_deviation)


def _offset(prior: float, absolute: _Rating) -> _Rating:
    """Back to an offset, after `glicko_step` has moved the absolute rating."""
    return _Rating(absolute.rating - prior, absolute.rating_deviation)

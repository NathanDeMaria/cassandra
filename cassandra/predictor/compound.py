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

Scoring a contest
-----------------

The result of a contest is the offense's EPA per play, **garbage time
adjusted** -- `GameEpa.home_weighted` / `.away_weighted`, each snap weighted
by how much the game was still in doubt when it happened. The game as it
was contested, not as it was run out. Minus what an average offense
averages (`epa_center`, a running mean of every offense seen so far), through
a logistic, and that is the score in [0, 1] that `glicko_step` takes:

    offense scores   sigmoid(offense_scale * (epa - center))
    defense scores   1 - sigmoid(defense_scale * (epa - center))

Per play, **not** times the game's snap count: an offense is rated on how
well it moved the ball, not on how often it got to, and a snap count would
credit the offense for a defense that kept handing it the ball back.

What the offense is measured *against* is the defense it faced, and that
happens where it does for the parent: `glicko_step` takes the expected score
from the gap between the two units and moves each by actual minus expected.
The center is not that comparison. It only says where 0.5 sits -- an
offense that played like the league's average offense scored half -- which
is what it takes for an average offense against an average defense to be a
draw, and for the anchors to be priors for both sides rather than for
both-sides-shifted-by-a-constant.

Each side reads the contest at its own sharpness, `offense_scale` and
`defense_scale`, so the two sides of one contest need not sum to 1: they
are two updates against the same expected score. There is no reason a
tenth of a point per snap has to be exactly as much evidence about a defense
as about an offense, and measured it isn't (below).

**Where the scale puts the units.** The parent's target for a game is
`sigmoid(margin / sigmoid_scale)` and its expected score is Glicko's
logistic of a rating gap, so a rating gap *is* an expected points margin at
whatever slope the league's fit found. A contest is half a game -- with a
team's unit rating the mean of its two sides, the two contests' gaps sum to
twice the gap between the teams' unit means -- so reading a per-play average
as points over a typical game (~80 snaps in ncaafb), through the parent's
squash, at twice the team slope, comes to a scale of `2 * 80 / 10 = 16`
logits per point of EPA per play. That is the setting under which the
children land on the parent's rating scale, and it is the default. It is
not a constraint: the search moves both scales, and a drift off the
parent's currency is absorbed by `unit_weight` at prediction.

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

Each side runs its own Glicko clock: `offense_initial_rd` and
`defense_initial_rd` (where a deviation starts, and the cap it grows back
to), `offense_weekly_rd_increase` / `defense_weekly_rd_increase`, and
`offense_season_rd_increase` / `defense_season_rd_increase`. All default to
the parent's. Offense is said to be the steadier of the two, and a steadier
side is a smaller initial deviation and smaller increases -- less to learn
at first, less to forget each week and each year. The search moves all six
apart; the parent's values are where it starts.

How the children speak
----------------------

The record and the units are two estimates of one quantity, each with a
deviation, so at prediction they are combined the way two noisy
measurements of one thing are: by precision.

    rating = (parent / rd_parent^2 + w * earned * unit_mean)
           / (1 / rd_parent^2 + w * earned)

`unit_mean` is `(offense + defense) / 2`; `earned` is its precision net of
what the pair started with -- 0 for a pair still at the initial deviations,
growing as they tighten -- because the prior is not evidence and the parent
has already counted it once; `w` is `unit_weight`, how far to trust a
unit's deviation against the parent's, with 1 taking both at face value.
Then the parent's own arithmetic -- home edge, matchup adjustments, Glicko's
logistic of the gap -- runs on that rating.

A team the index never had a play for has earned nothing and stays at its
record, which is most of an ncaafb schedule and all of it before 2006. That
matters more than it sounds: a blend that did not fade regressed every such
team toward its anchor and cost more on the games it couldn't see than it
earned on the ones it could (measured: +0.00019 overall against -0.00012 on
the EPA games).

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
skipped for mis-grouped weeks) against an index rebuilt with the weighted
pair, holding `glicko_full`'s fitted parameters and the same saved opponent
priors, so the parent is the baseline exactly and `unit_weight` 0 reproduces
its 0.158621. Every number is against that, and the EPA column is over the
20,860 games the index has for those seasons.

    unit_weight  offense_scale  defense_scale  parent_share   d brier   EPA games
           0.10             16              8           0.0  -0.000068   -0.000208
           0.10              8              8           0.0  -0.000057   -0.000155
           0.10             16             16           0.0  -0.000049   -0.000182
           0.10              8             16           0.0  -0.000042   -0.000139
           0.05              8              8           0.0  -0.000048   -0.000129
           0.20              8              8           0.0  -0.000002   -0.000021
           0.50             16             16           0.0  +0.000468   +0.000947
           1.00             16             16           0.0  +0.001268   +0.002667
           0.10              8              8           0.5  -0.000031   -0.000112
           0.10              8              8           1.0  +0.000028   +0.000024

Four readings. **A contest is a tenth of a game.** `unit_weight` 1 -- both
deviations at face value -- hands the units about half the rating and costs
0.0013; the optimum is near 0.1, which under precision weighting has a
plain meaning: as evidence about the team, a contest on EPA is worth about
a tenth of a result on the scoreboard. That is `glicko_blend`'s finding
arriving by a different road -- EPA is a noisier measurement of the thing
the scoreboard measures, worth the variance reduction of averaging in a
little of it and nothing beyond. **The anchor is the better prior**:
`parent_share` 0 beats 1 at every weight. Scoring a child against its
team's record makes it learn the part of the record the plays don't show,
and that part predicts less well than the plays do on their own. **The
offense reads sharper than the defense.** The best row scores the offense
at twice the defense's scale; the reverse split is the worst of the four
scale rows. A tenth of a point per snap is more evidence about the offense
that produced it than about the defense that allowed it, which is the
"offense is steadier" hypothesis showing up through the scale. **The
clocks don't say yet.** Single probes off the 8/8 row: giving the offense a
smaller season increase than the parent's (100 against 250) costs 0.00005
and a larger one (400) gains 0.00001; the defense's the other way round by
less; a weekly increase of 30 on both gains 0.00001. Tightening either
side's initial deviation to 300 costs 0.00001-0.00002. None of that is a
sweep, and none of it points the way the hypothesis predicts; the configs
search all six.

Against the same scoring, the blend the first draft used -- a linear
`(1 - w) * parent + w * units` with `w` faded by confidence -- reaches
-0.000030 at its best. Precision weighting is both the version with a
reason and the better one.

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
worse; fading unseen units out of the combination is what let the seeded
children be used at all.

**The signal is real.** Permute which game each `GameEpa` belongs to --
both sides kept together, every marginal preserved, only the correspondence
to the game played destroyed -- and replay the best row fifteen times:

    shuffles  null mean   vs off      real is
          15   0.159113  +0.000492   11.3 sd better, 0/15 matched it

Note the sign on the null, as `glicko_blend` notes it: a shuffled index is
*worse than no index at all*, by nearly nine times what the real one is
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
none, which is nothing earned and the parent alone. The one thing a release
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
from .glicko import DEFAULT_PREDICTION_SCALE, GlickoPredictor, _Rating, glicko_step
from .opponent_prior import OpponentPriorManager
from .qb_out import QbOutIndex
from .rest import DEFAULT_REST_ADVANTAGE
from .types import Matchup, Prediction, Rating, Unit, Units

# How far to trust a unit's deviation against the parent's, when the two are
# combined by precision at prediction. 1 takes both at face value, and is
# measured to cost 0.0013 brier: a unit's deviation shrinks at Glicko's rate
# per contest, and a contest is far weaker evidence about *winning* than a
# result is. A tenth is where the measurement sits -- see the module
# docstring -- and is the reading a fitted value carries: a contest is worth
# about a tenth of a game as evidence about the team. Not 0, which would
# make a hand-built one silently identical to the model it subclasses.
DEFAULT_UNIT_WEIGHT = 0.1

# How sharply a contest is scored: logit per point of EPA per play, one for
# each side. An offense a tenth of a point per snap above average scores
# `sigmoid(0.1 * scale)` -- 0.83 at 16 -- and the defense across from it
# scores one minus the same thing at *its* scale.
#
# 16 is what "read the per-play number as points over a typical game, through
# the parent's own squash, at twice the team slope" comes to for ncaafb
# (2 * 80 snaps / sigmoid_scale 10), which is the setting that puts the units
# on the parent's rating scale. That is where a search starts, and the
# measurement in the module docstring found 8 to 16 flat. The same number for
# both sides by default, because nothing has yet shown one side wants to be
# scored more sharply than the other; the search moves them apart.
DEFAULT_OFFENSE_SCALE = 16.0
DEFAULT_DEFENSE_SCALE = 16.0

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

    `offense_initial_rd` and `defense_initial_rd` are `None` for "the
    parent's `initial_rd`": a unit nobody has seen play is exactly as unknown
    as a team nobody has, and the parent's fit already says how unknown that
    is for the league. Two rather than one because the two sides need not be
    equally knowable -- offense is said to be the more stable of the two,
    which is a smaller deviation to start from and, since the initial value
    is also the cap the offseason grows back to, less to forget each year.
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
        prediction_scale: float = DEFAULT_PREDICTION_SCALE,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        unit_weight: float = DEFAULT_UNIT_WEIGHT,
        offense_scale: float = DEFAULT_OFFENSE_SCALE,
        defense_scale: float = DEFAULT_DEFENSE_SCALE,
        parent_share: float = DEFAULT_PARENT_SHARE,
        # The children's own Glicko clock, side by side: where a deviation
        # starts (and the cap it grows back to), how much it widens between
        # weeks, and how much between seasons. Each `None` is "the parent's",
        # which is what a hand-built one gets and where a search starts.
        offense_initial_rd: float | None = None,
        defense_initial_rd: float | None = None,
        offense_weekly_rd_increase: float | None = None,
        defense_weekly_rd_increase: float | None = None,
        offense_season_rd_increase: float | None = None,
        defense_season_rd_increase: float | None = None,
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
            prediction_scale=prediction_scale,
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            qb_out=qb_out,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            ratings=ratings,
            anchors=anchors,
        )
        if unit_weight < 0:
            # 0 is the off switch and stays legal. Below it a unit's evidence
            # counts *against* the parent's, which is a model that learns
            # backwards from its best-measured games.
            raise ValueError(f"unit_weight must be non-negative, got {unit_weight}")
        self._unit_weight = unit_weight
        self._offense_scale = validated_scale("offense_scale", offense_scale)
        self._defense_scale = validated_scale("defense_scale", defense_scale)
        self._parent_share = validated_fraction("parent_share", parent_share)
        self._offense_initial_rd = _or_parents(
            "offense_initial_rd", offense_initial_rd, initial_rd
        )
        self._defense_initial_rd = _or_parents(
            "defense_initial_rd", defense_initial_rd, initial_rd
        )
        # The increases may be 0 -- the parent's own is, in ncaafb's fit --
        # so they are checked for sign rather than through `validated_scale`.
        self._offense_weekly_rd_increase = _or_parents(
            "offense_weekly_rd_increase",
            offense_weekly_rd_increase,
            weekly_rd_increase,
            zero_ok=True,
        )
        self._defense_weekly_rd_increase = _or_parents(
            "defense_weekly_rd_increase",
            defense_weekly_rd_increase,
            weekly_rd_increase,
            zero_ok=True,
        )
        self._offense_season_rd_increase = _or_parents(
            "offense_season_rd_increase",
            offense_season_rd_increase,
            season_rd_increase,
            zero_ok=True,
        )
        self._defense_season_rd_increase = _or_parents(
            "defense_season_rd_increase",
            defense_season_rd_increase,
            season_rd_increase,
            zero_ok=True,
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
                _Rating(0.0, self._offense_initial_rd),
                _Rating(0.0, self._defense_initial_rd),
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

    def unit_information(self, team: str) -> float:
        """What this team's units have learned, as precision: 1 / rd^2, earned.

        The unit mean is `(offense + defense) / 2`, so its variance is a
        quarter of the two deviations' squared sum and its precision the
        reciprocal of that. *Earned* precision is that minus what the pair
        started with: 0 for a pair still at the initial deviations --
        nothing seen, or a release that never carried them -- and growing
        as they tighten. Net of the prior because the prior is not evidence:
        a unit nobody has seen is at its anchor, which the parent has
        already counted once.

        Glicko keeps exactly the numbers this needs, which is why the blend
        below can be a real one. A team the index never had a play for --
        most of an ncaafb schedule, all of it before 2006 -- earns nothing
        and stays at its record.
        """
        offsets = self._offsets(team)
        now = 4 / (
            offsets.offense.rating_deviation**2 + offsets.defense.rating_deviation**2
        )
        prior = 4 / (self._offense_initial_rd**2 + self._defense_initial_rd**2)
        return max(0.0, now - prior)

    def _blended_rating(self, team: str) -> float:
        """What a team is rated, for a prediction: record and units, combined.

        Two estimates of the same quantity, each with a deviation, combine
        by precision -- the inverse-variance weighting that is the standard
        answer to "I have two noisy measurements of one thing", and the one
        with a reason behind it. The parent brings `1 / rd^2`; the units
        bring what they have earned (`unit_information`), times
        `unit_weight`, which says how far to trust a unit's deviation
        against the parent's. At 1 the deviations are taken at face value;
        at 0 the units are silent; above 1 the units are trusted past what
        their deviations claim.
        """
        parent = self.get_rating(team)
        if not self._unit_weight:
            return parent.rating
        earned = self._unit_weight * self.unit_information(team)
        if not earned:
            return parent.rating
        parent_precision = 1 / parent.rating_deviation**2
        return (parent_precision * parent.rating + earned * self.unit_rating(team)) / (
            parent_precision + earned
        )

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
        return Prediction(team1_win_prob=self.win_prob(home_rating, away_rating))

    def update_game(self, game: Game) -> Prediction:
        """The parent's update, then the children's.

        The parents are read first and handed to `_update_units`, because
        `super().update_game` moves them and a unit's expected score belongs
        against the team its opponent was known to be when the contest was
        played -- the same "predict, then learn" ordering the parent itself
        follows. The home edge is read first for the same reason: the
        parent's update records the game in the rest ledger, after which
        both teams have just played and the gap that was there is gone. The
        children go second so the parent's `_update_rating` calls, which
        read both teams' ratings before writing either, see the state they
        expect.
        """
        home_parent = self.get_rating(game.home)
        away_parent = self.get_rating(game.away)
        home_edge = (
            0 if game.neutral_site else self._home_advantage
        ) + self.matchup_adjustment(game)
        prediction = super().update_game(game)
        self._update_units(game, home_parent, away_parent, home_edge)
        return prediction

    def _contest_score(self, epa_per_play: float, scale: float) -> float:
        """What an offense's game counts as, in [0, 1], at one side's sharpness.

        The garbage-time-adjusted average, `GameEpa.home_weighted` or
        `.away_weighted`, minus what an average offense averages -- the
        running `epa_center` -- through a logistic at `scale` logits per
        point of EPA per play. Per play, never times a snap count: an
        offense is rated on how well it moved the ball, not on how often it
        got to, and a snap count would credit the offense for a defense that
        kept handing it the ball back.

        The center is *not* what the offense is measured against. That is
        the opponent, and it happens in `glicko_step`: the expected score
        comes from the gap between this offense and the defense it faced,
        and the update is the actual score minus that. The center only says
        where 0.5 sits -- an offense that played like the league's average
        offense scored half, which is what it takes for an average offense
        against an average defense to be a draw and for the anchors to be
        priors for both sides rather than for both-sides-shifted.

        `scale` is the side's own: the offense reads its result at
        `offense_scale`, and the defense across from it reads one minus the
        same result at `defense_scale`. Two rather than one because there is
        no reason a tenth of a point per snap has to be exactly as much
        evidence about a defense as about an offense.
        """
        return 1 / (1 + math.exp(-scale * (epa_per_play - self.epa_center)))

    def _see(self, home: float, away: float) -> None:
        """Fold a game's two offenses -- the weighted reading -- into the center.

        Before the game is scored, not after, so the very first contest is
        centered on its own game rather than on 0 -- and after that it is
        one game in a thousand either way. It has already been played, so
        nothing here is read before it happened.
        """
        self._epa_sum += home + away
        self._epa_count += 2

    def _update_units(
        self,
        game: Game,
        home_parent: _Rating,
        away_parent: _Rating,
        home_edge: float,
    ) -> None:
        """Run the two unit contests, if the game has EPA to score them by.

        `home_parent` and `away_parent` are the parents *as they stood before
        the game*, passed in rather than re-read because `update_game` has
        already moved them. A unit's expected score is against the team its
        opponent was known to be when the contest was played, not the one it
        became by playing it. `home_edge` is read before the game too, for
        the rest ledger's sake.

        Each contest goes through `glicko_step` with both sides at prior plus
        offset, so the step sees absolute ratings on the parent's scale, and
        the offset stored is what the step returned minus the prior it was
        built on. The deviation is the child's own: the parent's is not
        folded in, because the parent's uncertainty is the parent's business
        and a child that inherited it would look twice as unsure as the
        evidence says.

        The home edge -- the parent's `home_advantage` plus the matchup
        adjustments -- goes on each contest whole. The parent's own update
        applies it the same way, and the two contests' edges sum to twice
        one edge exactly as their gaps sum to twice the team gap.
        """
        epa = self._game_epa.get(game.game_id)
        if epa is None or epa.home_weighted is None or epa.away_weighted is None:
            # No plays, or -- with the weighting on -- a side whose every snap
            # came with the game already decided. Nothing to rate either
            # contest on, so neither runs; the parent has already learned
            # from the score.
            return
        self._see(epa.home_weighted, epa.away_weighted)
        home_prior = self._prior(game.home, home_parent)
        away_prior = self._prior(game.away, away_parent)
        home = self._offsets(game.home)
        away = self._offsets(game.away)

        home_offense = _absolute(home_prior, home.offense)
        home_defense = _absolute(home_prior, home.defense)
        away_offense = _absolute(away_prior, away.offense)
        away_defense = _absolute(away_prior, away.defense)

        # Each side reads the contest at its own sharpness, so the two sides
        # of one contest need not sum to 1 -- they are two updates against
        # the same expected score, not two halves of one number.
        home_ball_o = self._contest_score(epa.home_weighted, self._offense_scale)
        home_ball_d = self._contest_score(epa.home_weighted, self._defense_scale)
        away_ball_o = self._contest_score(epa.away_weighted, self._offense_scale)
        away_ball_d = self._contest_score(epa.away_weighted, self._defense_scale)

        self._units[game.home] = _Units(
            offense=_offset(
                home_prior,
                glicko_step(home_offense, away_defense, home_ball_o, home_edge),
            ),
            defense=_offset(
                home_prior,
                glicko_step(home_defense, away_offense, 1 - away_ball_d, home_edge),
            ),
        )
        self._units[game.away] = _Units(
            offense=_offset(
                away_prior,
                glicko_step(away_offense, home_defense, away_ball_o, -home_edge),
            ),
            defense=_offset(
                away_prior,
                glicko_step(away_defense, home_offense, 1 - home_ball_d, -home_edge),
            ),
        )

    def _grow_unit_rd(
        self, units: _Units, offense_increase: float, defense_increase: float
    ) -> _Units:
        """Each side's deviation widened by its own increase, capped at its own initial."""
        return _Units(
            *(
                _Rating(
                    unit.rating,
                    min(cap, math.sqrt(unit.rating_deviation**2 + increase**2)),
                )
                for unit, increase, cap in zip(
                    units,
                    (offense_increase, defense_increase),
                    (self._offense_initial_rd, self._defense_initial_rd),
                )
            )
        )

    def pass_week(self) -> None:
        super().pass_week()
        self._units = {
            team: self._grow_unit_rd(
                units,
                self._offense_weekly_rd_increase,
                self._defense_weekly_rd_increase,
            )
            for team, units in self._units.items()
        }

    def _roll_over(self) -> None:
        """The parent's offseason, and the children's.

        A child's offset regresses toward 0 -- toward its prior, which is the
        team's anchor or its parent or a mix -- by the same
        `season_regression` the parent uses toward its anchor. Not through
        `regress`, whose target is the anchor itself: the offset is already
        measured from there. Then each side's deviation grows by its own
        season increase.
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
                self._offense_season_rd_increase,
                self._defense_season_rd_increase,
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
            "offense_scale": self._offense_scale,
            "defense_scale": self._defense_scale,
            "parent_share": self._parent_share,
            "offense_initial_rd": self._offense_initial_rd,
            "defense_initial_rd": self._defense_initial_rd,
            "offense_weekly_rd_increase": self._offense_weekly_rd_increase,
            "defense_weekly_rd_increase": self._defense_weekly_rd_increase,
            "offense_season_rd_increase": self._offense_season_rd_increase,
            "defense_season_rd_increase": self._defense_season_rd_increase,
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


def _or_parents(
    name: str, value: float | None, parents: float, zero_ok: bool = False
) -> float:
    """One of the children's clock settings: the value given, or the parent's.

    Checked the way the parent's own would be if the parent checked them:
    a deviation has to be positive, an increase only non-negative.
    """
    if value is None:
        return parents
    if value < 0 or (value == 0 and not zero_ok):
        raise ValueError(
            f"{name} must be {'non-negative' if zero_ok else 'positive'}, got {value}"
        )
    return value


def _absolute(prior: float, offset: _Rating) -> _Rating:
    """A child as `glicko_step` should see it: its prior plus its offset."""
    return _Rating(prior + offset.rating, offset.rating_deviation)


def _offset(prior: float, absolute: _Rating) -> _Rating:
    """Back to an offset, after `glicko_step` has moved the absolute rating."""
    return _Rating(absolute.rating - prior, absolute.rating_deviation)

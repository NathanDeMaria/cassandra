"""The margin model, with an offense and a defense per team rated on EPA per play.

`MarginGlickoPredictor` rates a team by its margins. This keeps that rating --
the *parent*, stepped exactly as the parent steps it -- and grows two *units*
under each team: an offense and a defense, each a Gaussian (mean, deviation)
pair in EPA per play, filtered on each offense's per-play average against the
defense it faced.

Why not `CompoundGlickoPredictor`
---------------------------------

That model has this shape and the wrong observation. It squashes a per-play
average through a logistic and hands it to `glicko_step`, which at its fitted
scales saturates about two contests in three; it rates on the garbage-time-
weighted reading, which split-half halves in reliability; and it centers on a
running mean over every season, which never caught the 2014 step in the
index. What an offline check says the units *should* be is simpler than any
of that: a per-season ridge,

    epa_off = center + edge + O[offense] - D[defense] + noise,  var ~ 1 / snaps

weighted by snaps, on the flat reading. Its top units on ncaafb are the ones
everyone would name (2019 LSU and Ohio State on offense, Clemson and Georgia
on defense; 2023 Michigan's defense; 2025 Indiana both ways), and it beats
raw EPA per play on the other half of a season by 0.05 R^2. The Kalman
filter below is that regression run forward one week at a time, and the
smoother (`passes`) re-walks the season toward it.

The contest
-----------

A game is two contests: the home offense against the away defense, and the
other way round. For one of them, with the offense at `O` (deviation `p`)
and the defense at `D` (`q`), both offsets in EPA per play:

    mu = center + (edge + anchor_off - anchor_def) * c / (2 S) + O - D
    R  = play_sd^2 / snaps                       the reading's noise
    v  = epa - mu
    O += p^2 / (p^2 + q^2 + R) * v               p^2 *= 1 - p^2 / (p^2 + q^2 + R)
    D -= q^2 / (p^2 + q^2 + R) * v               q^2 *= 1 - q^2 / (p^2 + q^2 + R)

which is the joint Kalman step on the two states, each side's marginal, with
the other side's deviation in the noise as the parent folds in an opponent's.
`S` is `points_per_epa` -- margin points per EPA per play of differential
between two offenses over a game, 75 on ncaafb -- and `c` is the parent's
`points_per_rating`, so `c / (2 S)` turns a rating gap into the per-play gap
each contest carries half of. The anchor term is what makes the anchors
priors for the units: an FBS offense against a D-II defense is expected to
move the ball, and a unit at offset 0 is a unit that did what its tier does.
The edge is the parent's own -- home advantage and matchup terms -- split
between the contests the same way.

The center is the league's average offense, per season: this season's
running mean, pulled toward last season's by `CENTER_PRIOR_SIDES` sides of
it, so a season opens where the last one closed and moves when the league
does. The step at 2014, where the feed became legible, is a season-to-season
move of 0.1 per play, and a mean over every season never catches it.

Offense and defense on their own clocks
---------------------------------------

Each side has its own starting deviation (the cap it grows back to), weekly
and seasonal increase, and offseason regression. For a win probability only
`O + D` counts -- a team's margin against another is `S` times the gap in
the two sums -- so if the two sides ran on one clock this would be a net EPA
rating with extra bookkeeping. The split is worth having only because the
two sides are known and forgotten at different rates, and measured, they
are: see below. It isn't the way round it is usually said. The offense is
the side that moves *during* a season (a quarterback is one injury from a
different unit) and the side that forgets more over the summer; a defense
is closer to fixed for the year and carries more of itself into the next.

How the units speak
-------------------

As `CompoundGlickoPredictor`'s do: by precision, at prediction, on the
parent's rating scale. A team's unit rating is

    anchor + (S / c) * (O + D)

with variance `(S / c)^2 (p^2 + q^2)`, and the rating predicted from is

    (parent / d^2 + w * earned * unit) / (1 / d^2 + w * earned)

where `earned` is the unit's precision net of what it started with -- 0 for
a team the index never had a play for, which stays at its record -- and `w`
is `unit_weight`. At `unit_weight` 0 this is `MarginGlickoPredictor` exactly,
game by game; the units never feed back into the parent's step and the
parent never sees an EPA number.

What a release carries
----------------------

`ratings` hands each side out on the team scale -- `anchor + 2 (S / c) O`,
so the mean of a team's two sides is its unit rating -- and `from_ratings`
takes them back. The center isn't carried, which only matters to a rebuilt
model that goes on updating; a prediction never reads it.

What it was measured to be worth
--------------------------------

Replayed on ncaafb 2002-2025 off the local season cache, on the fitted
`glicko_margin` parent (run 20260921; brier here 0.154604, since the replay
skips the opponent-prior warm-up), about forty hand probes. Brier against
`unit_weight` 0, which is the parent exactly:

    units                                all      EPA games  2014+ FBS  2014+ FCS  FBS-FCS
    first draft (weight 0.5, ridge
      clocks, play_sd 1.3)             -0.000049  -0.000343  -0.000321  +0.000403  -0.003946
    one clock for both sides, best     -0.000102  -0.000500  -0.000632  +0.000428  -0.003683
    split clocks (the defaults)        -0.000223  -0.000900  -0.000846  -0.000506  -0.003351

The split is worth as much again as the units on one clock, and it is what
turned FCS-vs-FCS from a loss into a gain. The probes that said which way
round, off the one-clock row: a weekly increase of 0.02 on the offense and
none on the defense gains 0.00037 on 2014+ FBS, the reverse costs 0.00025;
an offseason regression of 0.8 on the offense and 0.4 on the defense beats
the reverse; a season increase of 0.10 offense and 0.03 defense beats the
reverse by 0.0006 on FBS.

For scale: the best `CompoundGlickoPredictor` variant found on the same
cache gained 0.00004 on FBS-vs-FBS and the shipped one lost there. The
FBS-vs-FCS column is where the anchors' ladder was already known to be
off (the margin is over-dispersed across the tier gap), and a unit that
knows how an FCS offense moves against FBS defenses fixes a lot of it.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Self

from endgame.types import Game

from .adjustments import (
    DEFAULT_QB_OUT_PENALTY,
    DEFAULT_TRAVEL_ADVANTAGE,
    MatchupSources,
)
from .base_predictor import Anchor, validated_regression
from .blend import validated_scale
from .epa import EpaIndex
from .glicko import _Rating
from .margin_glicko import (
    DEFAULT_OBS_SD,
    DEFAULT_POINTS_PER_RATING,
    MarginGlickoPredictor,
    _normal_cdf,
)
from .opponent_prior import OpponentPriorManager
from .rest import DEFAULT_REST_ADVANTAGE
from .types import Matchup, Prediction, Rating, Unit, Units

#: Margin points per EPA per play of differential between the two offenses,
#: over a game. The slope of the margin on `home - away` per-play EPA over
#: ncaafb 2014-2025 is 73.5 (75.0 FBS-vs-FBS), which is the median 78 snaps
#: a side times a clip that understates blowouts a little.
DEFAULT_POINTS_PER_EPA = 75.0

#: Noise of one snap's clipped EPA about what the matchup predicted, so a
#: side's per-play average over `n` snaps reads with variance `play_sd^2 / n`.
#: The ridge's residuals put it between 1.1 (in hindsight, which understates
#: it) and 1.6; replayed, 1.0 beat 1.3 and 0.8 was worse again.
DEFAULT_PLAY_SD = 1.0

#: Where a unit's deviation starts, in EPA per play, and the cap it grows
#: back to. The within-division spread of adjusted units is 0.08 on FBS;
#: 0.14 leaves room for the conference the anchor doesn't see, and replayed
#: beat 0.10 and 0.18.
DEFAULT_OFFENSE_INITIAL_SD = 0.14
DEFAULT_DEFENSE_INITIAL_SD = 0.14

#: How much a unit's deviation widens each week. The offense moves in
#: season and the defense doesn't; see the module docstring.
DEFAULT_OFFENSE_WEEKLY_SD_INCREASE = 0.05
DEFAULT_DEFENSE_WEEKLY_SD_INCREASE = 0.0

#: How much a unit's deviation widens each offseason. The ridge's
#: year-to-year change is 0.069 offense and 0.063 defense; replayed, the
#: offense wants more room than that and the defense much less.
DEFAULT_OFFENSE_SEASON_SD_INCREASE = 0.10
DEFAULT_DEFENSE_SEASON_SD_INCREASE = 0.03

#: How far each offseason pulls a unit back to its anchor. The ridge says
#: both sides keep 0.55 of themselves; replayed, the offense keeps 0.2 and
#: the defense 0.8.
DEFAULT_OFFENSE_SEASON_REGRESSION = 0.8
DEFAULT_DEFENSE_SEASON_REGRESSION = 0.2

#: How far to trust a unit's deviation against the parent's at prediction.
#: The units' deviations are a Gaussian filter's, honest about EPA as a
#: reading of EPA; what they aren't honest about is how much of EPA is the
#: margin. 0.2 to 0.25 replayed best, 0.35 and 0.5 were worse.
DEFAULT_UNIT_WEIGHT = 0.2

#: How many sides of last season's average offense this season's center
#: starts with. A few weeks of games outweigh it.
CENTER_PRIOR_SIDES = 200


class _Sides(NamedTuple):
    """One team's offense and defense, as offsets from its anchor, in EPA per play."""

    offense: _Rating
    defense: _Rating


class _Contest(NamedTuple):
    """One offense against one defense, as the forward pass saw it.

    `base` is what the contest was expected to read before either unit's
    offset: the center, the anchors and the edge as they stood that day. Kept
    rather than re-derived for the reason `_Played` keeps the home edge: the
    center and the rest ledger have moved on by the time the smoother runs.
    """

    offense: str
    defense: str
    epa: float
    noise: float
    base: float


class UnitMarginGlickoPredictor(MarginGlickoPredictor):
    """The margin model, with offense and defense units rated on EPA per play.

    The parent's every method does what it did. What this adds is `_sides`,
    stepped from the EPA index after the parent has stepped on the margin,
    and read by `predict_game` as a second rating to combine with the first.
    See the module docstring for the step and the combination.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        home_advantage_slope: float = 0.0,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        obs_sd: float = DEFAULT_OBS_SD,
        nu: float | None = None,
        points_per_rating: float = DEFAULT_POINTS_PER_RATING,
        prediction_scale: float | None = None,
        # Both the parent's ratings and the units are re-walked this many
        # times a week; see `GlickoPredictor.passes`.
        passes: int = 1,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        unit_weight: float = DEFAULT_UNIT_WEIGHT,
        points_per_epa: float = DEFAULT_POINTS_PER_EPA,
        play_sd: float = DEFAULT_PLAY_SD,
        offense_initial_sd: float = DEFAULT_OFFENSE_INITIAL_SD,
        defense_initial_sd: float = DEFAULT_DEFENSE_INITIAL_SD,
        offense_weekly_sd_increase: float = DEFAULT_OFFENSE_WEEKLY_SD_INCREASE,
        defense_weekly_sd_increase: float = DEFAULT_DEFENSE_WEEKLY_SD_INCREASE,
        offense_season_sd_increase: float = DEFAULT_OFFENSE_SEASON_SD_INCREASE,
        defense_season_sd_increase: float = DEFAULT_DEFENSE_SEASON_SD_INCREASE,
        offense_season_regression: float = DEFAULT_OFFENSE_SEASON_REGRESSION,
        defense_season_regression: float = DEFAULT_DEFENSE_SEASON_REGRESSION,
        opponent_prior_manager: OpponentPriorManager | None = None,
        sources: MatchupSources | None = None,
        ratings: dict[str, _Rating] | None = None,
        sides: dict[str, _Sides] | None = None,
        # The center's state: last season's average offense (None before
        # any season with plays), and this season's sum and count of sides.
        # Carried so a loaded model keeps centering where the replay left it.
        epa_center_state: Sequence[float | None] = (None, 0.0, 0),
        anchors: Mapping[str, Anchor] | None = None,
        unanchored_seen: Sequence[float] = (0.0, 0.0, 0),
        game_epa: EpaIndex | None = None,
    ) -> None:
        super().__init__(
            league,
            home_advantage=home_advantage,
            home_advantage_slope=home_advantage_slope,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            obs_sd=obs_sd,
            nu=nu,
            points_per_rating=points_per_rating,
            prediction_scale=prediction_scale,
            passes=passes,
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            sources=sources,
            ratings=ratings,
            anchors=anchors,
            unanchored_seen=unanchored_seen,
        )
        if unit_weight < 0:
            # 0 is the off switch. Below it a unit's evidence counts against
            # the parent's, which is a model that learns backwards.
            raise ValueError(f"unit_weight must be non-negative, got {unit_weight}")
        self._unit_weight = unit_weight
        self._points_per_epa = validated_scale("points_per_epa", points_per_epa)
        self._play_sd = validated_scale("play_sd", play_sd)
        self._initial_sd = _Pair(
            validated_scale("offense_initial_sd", offense_initial_sd),
            validated_scale("defense_initial_sd", defense_initial_sd),
        )
        self._weekly_sd_increase = _Pair(
            _non_negative("offense_weekly_sd_increase", offense_weekly_sd_increase),
            _non_negative("defense_weekly_sd_increase", defense_weekly_sd_increase),
        )
        self._season_sd_increase = _Pair(
            _non_negative("offense_season_sd_increase", offense_season_sd_increase),
            _non_negative("defense_season_sd_increase", defense_season_sd_increase),
        )
        self._unit_regression = _Pair(
            validated_regression(offense_season_regression),
            validated_regression(defense_season_regression),
        )
        self._sides: dict[str, _Sides] = sides if sides is not None else {}
        previous, total, count = epa_center_state
        self._center_previous = None if previous is None else float(previous)
        self._center_sum = float(total or 0.0)
        self._center_count = int(count or 0)
        self._game_epa = game_epa or EpaIndex.for_league(league)
        # The smoother's ledger for the units, as the parent keeps one for
        # the ratings: where the season's sides started, and its contests.
        self._sides_preseason = dict(self._sides)
        self._contest_weeks: list[list[_Contest]] = []
        self._contests_this_week: list[_Contest] = []

    @property
    def unit_weight(self) -> float:
        """How far a unit's deviation is trusted against the parent's."""
        return self._unit_weight

    @property
    def points_per_epa(self) -> float:
        """Margin points per EPA per play of differential, over a game."""
        return self._points_per_epa

    @property
    def epa_center(self) -> float:
        """The league's average offense, in EPA per play, as this season has it.

        This season's running mean, with last season's as a prior worth
        `CENTER_PRIOR_SIDES` sides. 0 before anything has been seen.
        """
        if self._center_previous is None:
            if not self._center_count:
                return 0.0
            return self._center_sum / self._center_count
        return (self._center_previous * CENTER_PRIOR_SIDES + self._center_sum) / (
            CENTER_PRIOR_SIDES + self._center_count
        )

    @property
    def _rating_per_epa(self) -> float:
        """Rating points of team gap per EPA per play of `O + D`."""
        return self._points_per_epa / self.points_per_rating

    def _offsets(self, team: str) -> _Sides:
        return self._sides.get(
            team,
            _Sides(
                _Rating(0.0, self._initial_sd.offense),
                _Rating(0.0, self._initial_sd.defense),
            ),
        )

    def get_sides(self, team: str) -> _Sides:
        """This team's offense and defense offsets from its anchor, in EPA per play."""
        return self._offsets(team)

    def unit_rating(self, team: str) -> float:
        """The team as its units see it, on the parent's rating scale."""
        sides = self._offsets(team)
        return self.anchor(team) + self._rating_per_epa * (
            sides.offense.rating + sides.defense.rating
        )

    def unit_information(self, team: str) -> float:
        """What this team's units have learned, as rating precision net of the prior.

        The unit rating's variance is `(S / c)^2 (p^2 + q^2)`; its precision
        less what a pair at the initial deviations would have is what the
        units have earned, and 0 for a team the index never had a play for.
        """
        sides = self._offsets(team)
        scale = self._rating_per_epa**2
        now = 1 / (
            scale
            * (sides.offense.rating_deviation**2 + sides.defense.rating_deviation**2)
        )
        prior = 1 / (
            scale * (self._initial_sd.offense**2 + self._initial_sd.defense**2)
        )
        return max(0.0, now - prior)

    def _blended(self, team: str) -> _Rating:
        """The team's rating for a prediction: record and units, by precision.

        The deviation is the parent's, so the implied-Gaussian prediction's
        spread is the parent's too -- the units sharpen where a team sits,
        not how sure the margin model is of it.
        """
        parent = self.get_rating(team)
        if not self._unit_weight:
            return parent
        earned = self._unit_weight * self.unit_information(team)
        if not earned:
            return parent
        precision = 1 / parent.rating_deviation**2
        rating = (precision * parent.rating + earned * self.unit_rating(team)) / (
            precision + earned
        )
        return _Rating(rating, parent.rating_deviation)

    def predict_game(self, matchup: Matchup) -> Prediction:
        """The parent's prediction, on blended ratings.

        Written out rather than delegated because the parent reads
        `get_rating` directly; the arithmetic is the parent's, and
        `unit_weight` 0 reproducing it game by game is what guards that.
        """
        home = self._blended(matchup.home)
        away = self._blended(matchup.away)
        edge = self.home_edge(matchup) + self.matchup_adjustment(matchup)
        if not self._implied_prediction:
            return Prediction(
                team1_win_prob=self.win_prob(home.rating + edge, away.rating)
            )
        expected = self.points_per_rating * (home.rating + edge - away.rating)
        return Prediction(
            team1_win_prob=_normal_cdf(expected / self._margin_sd(home, away))
        )

    def update_game(self, game: Game) -> Prediction:
        """The parent's step on the margin, then the two contests on EPA.

        The edge is read first, before the parent records the game in the
        rest ledger, and the prediction is this model's, made before either
        step.
        """
        prediction = self.predict_game(game)
        edge = self.home_edge(game) + self.matchup_adjustment(game)
        # The parent's update makes the same prediction again and returns
        # it; the units haven't moved in between, so it is this one.
        super().update_game(game)
        self._update_sides(game, edge)
        return prediction

    def _update_sides(self, game: Game, edge: float) -> None:
        """Both contests, from the units as they stood before the game.

        The two contests touch disjoint units -- home offense and away
        defense, away offense and home defense -- so stepping them one after
        the other is stepping them together.
        """
        epa = self._game_epa.get(game.game_id)
        if epa is None or not epa.home_plays or not epa.away_plays:
            return
        center = self.epa_center
        self._center_sum += epa.home + epa.away
        self._center_count += 2
        per_play = self.points_per_rating / (2 * self._points_per_epa)
        gap = edge + self.anchor(game.home) - self.anchor(game.away)
        for offense, defense, reading, snaps, sign in (
            (game.home, game.away, epa.home, epa.home_plays, 1.0),
            (game.away, game.home, epa.away, epa.away_plays, -1.0),
        ):
            contest = _Contest(
                offense=offense,
                defense=defense,
                epa=reading,
                noise=self._play_sd**2 / snaps,
                base=center + sign * gap * per_play,
            )
            self._contests_this_week.append(contest)
            attack, guard = self._offsets(offense), self._offsets(defense)
            new_offense, new_defense = _contest_step(
                attack.offense, guard.defense, contest
            )
            self._sides[offense] = attack._replace(offense=new_offense)
            self._sides[defense] = self._offsets(defense)._replace(defense=new_defense)

    def pass_week(self) -> None:
        """The units' week, then the parent's, whose smoother re-walks both."""
        self._contest_weeks.append(self._contests_this_week)
        self._contests_this_week = []
        self._sides = self._aged_sides(self._sides, self._weekly_sd_increase)
        super().pass_week()

    def _aged_sides(
        self, sides: Mapping[str, _Sides], increase: "_Pair"
    ) -> dict[str, _Sides]:
        """Each side's deviation widened by its own increase, capped at its own initial.

        A side whose increase is 0 is left as it is rather than rebuilt: the
        defense's weekly increase is 0 by default, and this runs for every
        rated team every week of every smoothing pass.
        """
        offense, defense = increase
        return {
            team: _Sides(
                _widened(pair.offense, offense, self._initial_sd.offense)
                if offense
                else pair.offense,
                _widened(pair.defense, defense, self._initial_sd.defense)
                if defense
                else pair.defense,
            )
            for team, pair in sides.items()
        }

    def _smooth(self) -> None:
        """The parent's sweep, then the same sweep over the units.

        Each unit re-walked from preseason through its contests, the unit
        across from it held at the mean the previous pass ended with and its
        replayed deviation -- `GlickoPredictor._smooth`'s Gauss-Seidel,
        one level down. Only the means come out of it.

        Only the teams with a contest this season are replayed. Every other
        team's mean is its preseason one either way, and carrying the whole
        league's history through every week of every pass was most of what
        the smoother cost.
        """
        super()._smooth()
        played = {
            team
            for week in self._contest_weeks
            for contest in week
            for team in (contest.offense, contest.defense)
        }
        if not played:
            return
        settled = self._sides
        unseen = _Sides(
            _Rating(0.0, self._initial_sd.offense),
            _Rating(0.0, self._initial_sd.defense),
        )
        replay = {team: self._sides_preseason.get(team, unseen) for team in played}

        for week in self._contest_weeks:
            for contest in week:
                attack, guard = replay[contest.offense], replay[contest.defense]
                held_defense = _Rating(
                    settled[contest.defense].defense.rating,
                    guard.defense.rating_deviation,
                )
                held_offense = _Rating(
                    settled[contest.offense].offense.rating,
                    attack.offense.rating_deviation,
                )
                new_offense, _ = _contest_step(attack.offense, held_defense, contest)
                _, new_defense = _contest_step(held_offense, guard.defense, contest)
                replay[contest.offense] = attack._replace(offense=new_offense)
                replay[contest.defense] = replay[contest.defense]._replace(
                    defense=new_defense
                )
            replay = self._aged_sides(replay, self._weekly_sd_increase)

        smoothed = dict(self._sides)
        for team in played:
            pair = self._sides[team]
            smoothed[team] = _Sides(
                pair.offense._replace(rating=replay[team].offense.rating),
                pair.defense._replace(rating=replay[team].defense.rating),
            )
        self._sides = smoothed

    def _roll_over(self) -> None:
        """The parent's offseason, and each side's own.

        An offset regresses toward 0 -- its anchor -- by its side's
        regression, then its deviation grows by its side's increase. The
        center closes the season: its mean becomes next season's prior.
        """
        super()._roll_over()
        keep = _Pair(
            1 - self._unit_regression.offense, 1 - self._unit_regression.defense
        )
        self._sides = self._aged_sides(
            {
                team: _Sides(
                    pair.offense._replace(rating=keep.offense * pair.offense.rating),
                    pair.defense._replace(rating=keep.defense * pair.defense.rating),
                )
                for team, pair in self._sides.items()
            },
            self._season_sd_increase,
        )
        if self._center_count:
            self._center_previous = self._center_sum / self._center_count
        self._center_sum, self._center_count = 0.0, 0
        self._sides_preseason = dict(self._sides)
        self._contest_weeks = []
        self._contests_this_week = []

    def state_dict(self) -> dict[str, Any]:
        """The parent's state, the unit knobs, the center and the sides.

        The EPA index stays out, as it does for every play-by-play model.
        """
        return {
            **super().state_dict(),
            "unit_weight": self._unit_weight,
            "points_per_epa": self._points_per_epa,
            "play_sd": self._play_sd,
            "offense_initial_sd": self._initial_sd.offense,
            "defense_initial_sd": self._initial_sd.defense,
            "offense_weekly_sd_increase": self._weekly_sd_increase.offense,
            "defense_weekly_sd_increase": self._weekly_sd_increase.defense,
            "offense_season_sd_increase": self._season_sd_increase.offense,
            "defense_season_sd_increase": self._season_sd_increase.defense,
            "offense_season_regression": self._unit_regression.offense,
            "defense_season_regression": self._unit_regression.defense,
            "epa_center_state": [
                self._center_previous,
                self._center_sum,
                self._center_count,
            ],
            "sides": {
                team: [
                    pair.offense.rating,
                    pair.offense.rating_deviation,
                    pair.defense.rating,
                    pair.defense.rating_deviation,
                ]
                for team, pair in self._sides.items()
            },
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        params = dict(data)
        params["sides"] = {
            team: _Sides(_Rating(s[0], s[1]), _Rating(s[2], s[3]))
            for team, s in params.pop("sides", {}).items()
        }
        return super().from_state_dict(params)

    @property
    def ratings(self) -> dict[str, Rating]:
        """The parent's ratings, each carrying its two sides where there are any.

        On the team scale, `anchor + 2 (S / c) offset`, so a team's two
        sides average to its unit rating and a 1650 offense reads like a
        1650 team. Teams the index never had a play for carry none.
        """
        return {
            team: rating._replace(
                units=self._absolute_units(team) if team in self._sides else None
            )
            for team, rating in super().ratings.items()
        }

    def _absolute_units(self, team: str) -> Units:
        anchor = self.anchor(team)
        scale = 2 * self._rating_per_epa
        pair = self._sides[team]
        return Units(
            Unit(
                anchor + scale * pair.offense.rating,
                scale * pair.offense.rating_deviation,
            ),
            Unit(
                anchor + scale * pair.defense.rating,
                scale * pair.defense.rating_deviation,
            ),
        )

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        """The parent's rebuild, then the sides put back as offsets from this model's anchors."""
        predictor = super().from_ratings(league, ratings, **params)
        scale = 2 * predictor._rating_per_epa
        for team, rating in ratings.items():
            if rating.units is None:
                continue
            anchor = predictor.anchor(team)
            offense, defense = rating.units
            predictor._sides[team] = _Sides(
                _Rating((offense.rating - anchor) / scale, offense.rd / scale),
                _Rating((defense.rating - anchor) / scale, defense.rd / scale),
            )
        return predictor


class _Pair(NamedTuple):
    """One clock setting, per side."""

    offense: float
    defense: float


def _contest_step(
    offense: _Rating, defense: _Rating, contest: _Contest
) -> tuple[_Rating, _Rating]:
    """One contest's Kalman step, both units from the same innovation.

    Each unit's gain is its own variance over the innovation's, which is
    both variances and the reading's noise: the joint step's marginals, the
    cross-covariance it would build dropped as Glicko drops it.
    """
    p2 = offense.rating_deviation**2
    q2 = defense.rating_deviation**2
    total = p2 + q2 + contest.noise
    innovation = contest.epa - (contest.base + offense.rating - defense.rating)
    return (
        _Rating(
            offense.rating + p2 / total * innovation,
            math.sqrt(p2 * (1 - p2 / total)),
        ),
        _Rating(
            defense.rating - q2 / total * innovation,
            math.sqrt(q2 * (1 - q2 / total)),
        ),
    )


def _widened(unit: _Rating, increase: float, cap: float) -> _Rating:
    return _Rating(
        unit.rating, min(cap, math.sqrt(unit.rating_deviation**2 + increase**2))
    )


def _non_negative(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value

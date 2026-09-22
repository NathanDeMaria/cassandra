import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Self

from endgame.types import Game

from ..scoring import DEFAULT_SIGMOID_SCALE, get_scoring_function
from .adjustments import (
    DEFAULT_QB_OUT_PENALTY,
    DEFAULT_TRAVEL_ADVANTAGE,
    MatchupAdjustments,
    MatchupSources,
    resolved_sources,
)
from .base_predictor import (
    MEAN_RATING,
    Anchor,
    Predictor,
    resolved_anchors,
    validated_regression,
)
from .blend import validated_scale
from .opponent_prior import OpponentPriorManager
from .rest import DEFAULT_REST_ADVANTAGE
from .types import Matchup, Prediction, Rating


class _Rating(NamedTuple):
    rating: float
    rating_deviation: float


class _Played(NamedTuple):
    """One game as the forward update saw it, for the smoother to replay.

    The actual and the home edge are kept rather than re-derived: both read
    state that has moved on by the time the season is replayed -- rest is a
    differential built from each team's *last* game, and a replay of week 1
    after week 6 would price week 1 against week 5's rest.
    """

    home: str
    away: str
    actual: float
    home_adjustment: float


_Q = math.log(10) / 400

#: The widest expected-score exponent `glicko_step` takes: `10**100` puts
#: the expectation at 1e-100, which is 0 for every purpose but the division
#: by `p (1 - p)` behind it. Run 20260921-050501 hit that division at 0 on
#: one ncaafb probe with `sigmoid_scale` near its floor.
_MAX_EXPONENT = 100.0

#: How many unfiled teams' seasons the running estimate needs before it is
#: used over the league mean. One team's rating is that team; a handful is a
#: population.
MIN_UNANCHORED_SEEN = 5

#: Anchor points per unit of `home_advantage_slope`: the slope is quoted per
#: 400 anchor points -- one Elo decade, and roughly the FBS-to-FCS gap on
#: the ncaafb ladder -- so its value is a readable "how much more edge does
#: the tier above get".
_ANCHOR_SLOPE_SCALE = 400.0

# The rating gap that reads as a 10-to-1 favorite when a game is *predicted*.
# Elo's 400, which is also the scale the update learns at (`_Q`), so a model
# that never named one predicts exactly as it always has.
DEFAULT_PREDICTION_SCALE = 400.0


class GlickoPredictor(Predictor):
    """Glicko rating system.

    https://en.wikipedia.org/wiki/Glicko_rating_system
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        # How much more home advantage a team gets per 400 anchor points
        # above the league mean, in rating units. The one knob a per-tier
        # home advantage needs, and the anchors are the tiers the model
        # already has: `division_anchors.py` fits a rating per division (and
        # a conference offset inside it), so a team's anchor *is* its level,
        # and a home advantage that scales with it is one per division
        # without a classification lookup at prediction time or a parameter
        # per tier to search.
        #
        # Measured on ncaafb `glicko_full`, own-side residual at home minus
        # away, pooled by the home team's anchor: -0.3 points at anchors
        # near 1200 (D-III), +1.0 at 1875-2100 (FBS), against one constant
        # of 3.06 -- a weighted slope of +0.64 points per 400 anchor points
        # (se 0.11) across teams, and flat *within* FBS (+0.04 +/- 0.65),
        # so it is the tier that carries it and a line through the tiers is
        # the right shape. See `reasoning/residual_sweep_2026-09.md`.
        # Defaulted to 0 so every model published before this existed
        # replays exactly as it did; a league with no anchors (nfl) has
        # every team at the mean and the knob does nothing.
        home_advantage_slope: float = 0.0,
        k: float = 65,
        # increase is 'c' in the paper
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        scoring_method: str = "binary",
        # Only `scoring_method="sigmoid"` reads this; see
        # `get_scoring_function`. Accepted whatever the method is so a search
        # can move the two together rather than needing the categorical
        # resolved first.
        sigmoid_scale: float = DEFAULT_SIGMOID_SCALE,
        # How far apart two ratings have to be for one side to be a 10-to-1
        # favorite. Elo's answer is 400 and the *update* keeps it (`_Q`), so
        # this only changes how a rating gap is read as a win probability,
        # not how a game moves the ratings.
        #
        # Its own knob because the two scales are not the same question once
        # `scoring_method` is "sigmoid". The update then chases the expected
        # *margin score* of a game, which a dominant team never scores a full
        # 1.0 on, so the ratings settle closer together than the win
        # probabilities they are read as would need. `reasoning/
        # prediction_scale.ipynb` shows what that looked like on ncaafb:
        # calibrated in the first four games of a season, then about five
        # points under-confident at both ends of the scale for the rest of
        # it. The search had one lever to widen a prediction with -- the
        # deviations, which set how far a game moves a rating -- and it
        # pulled it: an `initial_rd` above 400, which is also why the first
        # game of a season taught twice what the sixth did. This is the lever
        # it wanted.
        prediction_scale: float = DEFAULT_PREDICTION_SCALE,
        # The three matchup terms. Each 0 is off, which is what every model
        # published before them replayed with. See `MatchupAdjustments`.
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        # How many times each week's games are rated. 1 is Glicko as a
        # filter: a result is priced once, against what the opponent's
        # rating was that day, and never revisited. Above 1 the season so
        # far is re-rated from its preseason state after every week, each
        # team's own rating evolving from preseason while every opponent is
        # priced at the mean the previous pass ended with -- so a week-1
        # win over a team that week 2 exposed is worth what it turned out
        # to be worth, and a preseason favourite that was blown out by a
        # scrub gives back the credit it took. See `_smooth`.
        #
        # Worth +0.0006 brier on ncaafb at 3 passes, over 24 of 25 seasons
        # and every week of the season, with the 2026-09-12 fit's other
        # parameters held where the filter tuned them; a rounding error on
        # nfl, where 17 games in a tight league leave the filter little to
        # learn from the second look. Defaulted to 1 so every model
        # published before this existed replays exactly as it did.
        passes: int = 1,
        opponent_prior_manager: OpponentPriorManager | None = None,
        # Where the matchup terms read their facts. Defaulted rather than
        # required, like the prior manager: a replay wants the league's own,
        # and a caller who knows something it can't derive -- an injury
        # report, a fixture's rest -- hands over a bundle that says so.
        sources: MatchupSources | None = None,
        ratings: dict[str, _Rating] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        # What teams with no anchor have turned out to be: the sum, sum of
        # squares and count of their end-of-season ratings, gathered as the
        # replay goes. Not a knob -- state, carried so a loaded model enters
        # an unfiled team where the replay that built it would. See
        # `unanchored_prior`.
        unanchored_seen: Sequence[float] = (0.0, 0.0, 0),
    ) -> None:
        super().__init__(league)
        self._anchors = resolved_anchors(league, anchors)
        total, squares, count = unanchored_seen
        self._unanchored_seen = (float(total), float(squares), int(count))
        self._season_regression = validated_regression(season_regression)
        self._home_advantage = home_advantage
        self._home_advantage_slope = home_advantage_slope
        self._k = k
        self._weekly_rd_increase = weekly_rd_increase
        self._season_rd_increase = season_rd_increase
        self._initial_rd = initial_rd
        self._scoring_method = scoring_method
        self._sigmoid_scale = validated_scale("sigmoid_scale", sigmoid_scale)
        self._score = get_scoring_function(scoring_method, self._sigmoid_scale)
        self._prediction_scale = validated_scale("prediction_scale", prediction_scale)
        self._adjustments = MatchupAdjustments(
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            sources=resolved_sources(league, sources),
        )

        self._prior_manager = opponent_prior_manager or OpponentPriorManager(
            league, model=self.__class__.__name__
        )
        if ratings is not None:
            self._ratings: dict[str, _Rating] = ratings
        else:
            prior_ratings = self._prior_manager.get_ratings()
            self._ratings = {
                team: _Rating(rating, self._initial_rd)
                for team, rating in prior_ratings.items()
            }
        self._passes = validated_passes(passes)
        # What the smoother replays from and over: the ratings the season
        # opened with, and the season's games so far, a list per week. In-
        # season state, not part of `state_dict` -- a restored predictor is
        # read, not trained further (see `serving.release`).
        self._preseason = dict(self._ratings)
        self._weeks: list[list[_Played]] = []
        self._this_week: list[_Played] = []

    def home_edge(self, matchup: Matchup) -> float:
        """Rating points the home side gets for being at home, before the matchup terms.

        0 at a neutral site. Otherwise the league constant plus the anchor
        slope -- see `home_advantage_slope` -- read at the home team's
        anchor for the season in hand, so a program that moved up gets the
        tier it moved to. Its own method because three places take the edge
        (the prediction, the update and `CompoundGlickoPredictor`'s unit
        contests) and they have to agree to the rating point.
        """
        if matchup.neutral_site:
            return 0.0
        edge = self._home_advantage
        if self._home_advantage_slope:
            edge += (
                self._home_advantage_slope
                * (self.anchor(matchup.home) - MEAN_RATING)
                / _ANCHOR_SLOPE_SCALE
            )
        return edge

    def predict_game(self, matchup: Matchup) -> Prediction:
        home_rating = self.get_rating(matchup.home)
        adjusted_home_rating = home_rating.rating + self.home_edge(matchup)
        # Outside the neutral-site guard on purpose: nobody is at home in a
        # bowl and both teams still arrived on different rest, and neither
        # has its quarterback back because the game is neutral. The travel
        # term zeroes itself there -- see `MatchupAdjustments.travel_points`.
        adjusted_home_rating += self.matchup_adjustment(matchup)
        away_rating = self.get_rating(matchup.away)
        return Prediction(
            team1_win_prob=self.win_prob(adjusted_home_rating, away_rating.rating)
        )

    def win_prob(self, home_rating: float, away_rating: float) -> float:
        """How a rating gap reads as a win probability, at `prediction_scale`.

        Here rather than inline in `predict_game` because a subclass that
        predicts from a different rating -- `CompoundGlickoPredictor` blends
        the record's with the units' -- still has to read the gap the way
        the parent does, or its `unit_weight` 0 stops being the parent.
        """
        return 1 / (1 + 10 ** ((away_rating - home_rating) / self._prediction_scale))

    def update_game(self, game: Game) -> Prediction:
        prediction = self.predict_game(game)
        home_rating = self.get_rating(game.home)
        away_rating = self.get_rating(game.away)
        actual = self._actual(game)
        # The same edge the prediction gave the home side, so the update
        # measures the result against what the matchup made likely rather
        # than against the bare ratings. Leaving the matchup terms out here
        # charged a backup's loss to the rating in full and gave nothing to
        # a team that beat a rested one -- the prediction knew the odds were
        # different, and the update pretended it didn't.
        home_adj = self.home_edge(game) + self.matchup_adjustment(game)

        self._update_rating(
            game.home, home_rating, away_rating, actual, home_adjustment=home_adj
        )
        self._update_rating(
            game.away, away_rating, home_rating, 1 - actual, home_adjustment=-home_adj
        )
        self._this_week.append(_Played(game.home, game.away, actual, home_adj))

        self._prior_manager.add_game(game)
        # After the prediction, so a game never contributes to its own rest.
        self._adjustments.record(game)
        return prediction

    def _actual(self, game: Game) -> float:
        """What this game counts as for the home team, in [0, 1].

        1 for a win, 0 for a loss, and whatever `scoring_method` says in
        between. Its own method because it is the one part of the update
        that isn't Glicko: everything around it is the rating arithmetic,
        while this is the question of what the game is evidence *of*. A
        subclass that answers it differently -- `ControlGlickoPredictor`
        blends in how much of the game each team spent winning it -- replaces
        this and inherits the rest.
        """
        return self._score(game)

    def _update_rating(
        self,
        team: str,
        my_rating: _Rating,
        opp_rating: _Rating,
        score: float,
        home_adjustment: float,
    ) -> None:
        self._ratings[team] = glicko_step(my_rating, opp_rating, score, home_adjustment)

    def get_rating(self, team: str) -> _Rating:
        # See EloPredictor.get_rating. The rd stays `initial_rd` for a team
        # with an anchor: knowing which division a team plays in says where
        # its rating starts, not how sure we are of it, and folding the
        # anchor into the deviation would make D-III teams look better-
        # measured than anyone has evidence for. A team with *no* anchor is
        # less sure than that -- see `unanchored_rd`.
        found = self._ratings.get(team)
        if found is not None:
            return found
        if team in self._anchors:
            return _Rating(self.anchor(team), self._initial_rd)
        return _Rating(self.unanchored_prior(), self.unanchored_rd())

    def unanchored_prior(self) -> float:
        """Where a team nobody filed a tier for enters: where the last ones ended up.

        The registry classifies every FBS and FCS program, so a team it has
        no tier for is almost always a D-II or D-III program it hasn't
        merged yet, or an exhibition opponent -- and either way not an
        average team. On ncaafb 133 such teams entered at the league mean of
        1500 and lost their first game by 33 points more than predicted,
        their second by 10, and took eight games to be rated where they
        belonged. Entering them at 1100 was worth 0.0004 brier and 0.036
        points of margin over the whole league, on 3% of its team-games.

        Rather than a number to search, this is measured as the replay goes:
        at every rollover the end-of-season rating of each unanchored team
        that played is folded into a running mean, and the next unfiled team
        enters there. The first such teams of a replay enter at the league
        mean, which is the honest answer with nothing seen; the estimate
        settles within a few seasons. `regress` pulls toward the same
        number, so an unfiled team's offseason takes it back to what unfiled
        teams are, not to the middle of the league.
        """
        total, _, count = self._unanchored_seen
        if count < MIN_UNANCHORED_SEEN:
            return super().unanchored_prior()
        return total / count

    def unanchored_rd(self) -> float:
        """How unsure to be of a team nobody filed a tier for.

        `initial_rd` is how unsure the league is of a team whose tier it
        knows. An unfiled team's tier is itself a guess, so its deviation
        carries the spread of where unfiled teams have ended up on top of
        that, in quadrature -- the two uncertainties are about different
        things. A wider deviation is a bigger first step, which is what lets
        a Kalman-style update (`MarginGlickoPredictor`) put a scrimmage
        opponent where it belongs in a game or two; the sigmoid target caps
        how far one game can move a rating whatever the deviation, so the
        parent gains less from it.
        """
        total, squares, count = self._unanchored_seen
        if count < MIN_UNANCHORED_SEEN:
            return self._initial_rd
        variance = max(0.0, squares / count - (total / count) ** 2)
        return math.sqrt(self._initial_rd**2 + variance)

    def _note_unanchored(self) -> None:
        """Fold this season's unfiled teams into the running estimate.

        Only teams that played this season, read off the smoother's ledger,
        so a team that appeared once in 2007 isn't counted again every
        offseason; and their ratings as they stand before the rollover
        regresses them.
        """
        played = {
            team
            for week in (*self._weeks, self._this_week)
            for game in week
            for team in (game.home, game.away)
        }
        total, squares, count = self._unanchored_seen
        for team in played:
            if team in self._anchors:
                continue
            rating = self._ratings[team].rating
            total += rating
            squares += rating**2
            count += 1
        self._unanchored_seen = (total, squares, count)

    def pass_week(self) -> None:
        self._weeks.append(self._this_week)
        self._this_week = []
        self._ratings = self._aged(self._ratings, self._weekly_rd_increase)
        for _ in range(self._passes - 1):
            self._smooth()

    def _aged(
        self, ratings: Mapping[str, _Rating], increase: float
    ) -> dict[str, _Rating]:
        """Every deviation widened by `increase`, capped at `initial_rd`."""
        return {
            team: _Rating(
                rating.rating,
                min(
                    self._initial_rd,
                    math.sqrt(rating.rating_deviation**2 + increase**2),
                ),
            )
            for team, rating in ratings.items()
        }

    def _smooth(self) -> None:
        """One more pass over the season so far; see `passes`.

        A Gauss-Seidel sweep, in effect: each team's rating is re-walked
        from preseason through its games in order, and the opponent it is
        measured against in each is held at the mean the previous pass
        finished with. Only the means come out of it. The deviations are
        the forward pass's, kept as they stand -- `glicko_step` shrinks a
        deviation every time it sees a game, and a pass that re-saw the
        whole season would leave every team looking measured three times
        over. The opponent's deviation inside the replay is its replayed
        one, so `g` discounts an early opponent the way the filter did.

        Passes converge quickly and wobble a little past three (five was a
        hair worse on ncaafb), which is a fixed-point iteration doing what
        they do; nothing here damps it because nothing has needed to.
        """
        settled = {team: rating.rating for team, rating in self._ratings.items()}
        replay = dict(self._preseason)

        def rating_of(team: str) -> _Rating:
            return replay.get(team, _Rating(self.anchor(team), self._initial_rd))

        for week in self._weeks:
            for played in week:
                home, away = rating_of(played.home), rating_of(played.away)
                replay[played.home] = glicko_step(
                    home,
                    _Rating(
                        settled.get(played.away, away.rating), away.rating_deviation
                    ),
                    played.actual,
                    played.home_adjustment,
                )
                replay[played.away] = glicko_step(
                    away,
                    _Rating(
                        settled.get(played.home, home.rating), home.rating_deviation
                    ),
                    1 - played.actual,
                    -played.home_adjustment,
                )
            replay = self._aged(replay, self._weekly_rd_increase)

        self._ratings = {
            team: _Rating(
                replay[team].rating if team in replay else rating.rating,
                rating.rating_deviation,
            )
            for team, rating in self._ratings.items()
        }

    def _roll_over(self) -> None:
        self._note_unanchored()
        self._ratings = {
            team: _Rating(
                self.regress(team, rating.rating),
                min(
                    self._initial_rd,
                    math.sqrt(rating.rating_deviation**2 + self._season_rd_increase**2),
                ),
            )
            for team, rating in self._ratings.items()
        }
        # The season the smoother replays from starts here: after the
        # regression and the offseason widening, with no games yet.
        self._preseason = dict(self._ratings)
        self._weeks = []
        self._this_week = []

    def postrun_callback(self) -> None:
        self._prior_manager.save(
            {team: rating.rating for team, rating in self._ratings.items()}
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "league": self._league,
            "home_advantage": self._home_advantage,
            "home_advantage_slope": self._home_advantage_slope,
            "k": self._k,
            "weekly_rd_increase": self._weekly_rd_increase,
            "season_rd_increase": self._season_rd_increase,
            "initial_rd": self._initial_rd,
            "scoring_method": self._scoring_method,
            "sigmoid_scale": self._sigmoid_scale,
            "prediction_scale": self._prediction_scale,
            "rest_advantage": self._adjustments.rest_advantage,
            "travel_advantage": self._adjustments.travel_advantage,
            "qb_out_penalty": self._adjustments.qb_out_penalty,
            "season_regression": self._season_regression,
            "passes": self._passes,
            "unanchored_seen": list(self._unanchored_seen),
            "ratings": {
                team: [r.rating, r.rating_deviation]
                for team, r in self._ratings.items()
            },
            "anchors": self._anchors,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        params = dict(data)
        ratings = params.pop("ratings")
        return cls(
            **params,
            ratings={team: _Rating(r[0], r[1]) for team, r in ratings.items()},
        )

    @property
    def ratings(self) -> dict[str, Rating]:
        return {
            team: Rating(r.rating, r.rating_deviation)
            for team, r in self._ratings.items()
        }

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        # Built empty first so `initial_rd` comes from params (or the
        # constructor default) rather than being repeated here.
        predictor = cls(league, ratings={}, **params)
        predictor._ratings = {
            team: _Rating(
                r.rating,
                # A rating with no rd is one we've never seen play, which is
                # exactly what get_rating hands back for an unknown team.
                r.rd if r.rd is not None else predictor._initial_rd,
            )
            for team, r in ratings.items()
        }
        return predictor


def validated_passes(passes: int) -> int:
    """`passes` as an int of at least 1, which is the filter.

    A search hands over a float, so 2.0 is 2 rather than a type error, but
    2.5 passes is not a thing and neither is 0: the first pass is the update
    itself, and a model with none of it isn't rating anything.
    """
    if passes != int(passes) or int(passes) < 1:
        raise ValueError(f"passes must be a whole number of at least 1, not {passes}")
    return int(passes)


def glicko_step(
    my_rating: _Rating,
    opp_rating: _Rating,
    score: float,
    home_adjustment: float,
) -> _Rating:
    """One Glicko update: where `my_rating` lands after `score` against `opp_rating`.

    A pure function rather than a method because the arithmetic is the same
    whatever the two ratings stand for. `GlickoPredictor` runs a team through
    it against a team; `CompoundGlickoPredictor` runs an offense through it
    against a defense. One copy, so the two cannot drift apart by a sign.

    `home_adjustment` is added to `my_rating` before the expected score is
    taken: positive for the side at home, negative for the side that isn't.
    """
    g_opp = _g(opp_rating.rating_deviation)
    exponent = g_opp * (opp_rating.rating - (my_rating.rating + home_adjustment)) / 400
    # A gap the arithmetic can't hold -- a corner probe pricing a point at
    # thousands of rating units -- is a certainty either way. Held just off 0
    # and 1 so `d2` stays finite and the step is the limit it tends to: no
    # change to the deviation, and the rating moved by the full surprise.
    exponent = min(max(exponent, -_MAX_EXPONENT), _MAX_EXPONENT)
    expected_score = 1 / (1 + 10**exponent)
    d2 = 1 / (_Q**2 * g_opp**2 * expected_score * (1 - expected_score))
    rd_inv_sq = 1 / my_rating.rating_deviation**2
    rd_inv_plus_d2 = rd_inv_sq + 1 / d2
    rd_new = math.sqrt(1 / rd_inv_plus_d2)
    rating_new = my_rating.rating + (_Q / rd_inv_plus_d2) * g_opp * (
        score - expected_score
    )
    return _Rating(rating_new, rd_new)


def _g(rating_deviation: float) -> float:
    return 1 / math.sqrt(1 + 3 * _Q**2 * rating_deviation**2 / math.pi**2)

"""A margin distribution read straight off a team's parts.

`UnitMarginGlickoPredictor` keeps two estimates of the same thing -- a record
stepped on margins and a pair of units stepped on plays, each learning from
the same games -- and averages them by precision at prediction, through a
`unit_weight` that runs away when it isn't capped, then reads a win
probability off a logistic whose scale was searched apart from either
filter's noise. This keeps one estimate, in four parts per team:

    L   level, on the rating scale the anchors are on: where the team opened
        the season, and what this season's games have moved it by
    O   offense, EPA per play: this season's deviation from what L implies
    D   defense, the same, signed so that more is better
    R   rest, points: what the margins said that the plays didn't --
        special teams, turnovers, finishing, and the share of the big plays
        a success-rate reading leaves out

and predicts the margin from their sum:

    mu    = c (L_h - L_a + edge) + S (O_h + D_h - O_a - D_a) + R_h - R_a
    sd^2  = game_sd^2 + uncertainty_scale * c^2 (dL_h^2 + dL_a^2)

with the win probability `P(margin > 0)` of that Normal. `c` is points per
rating unit and `S` points per EPA per play, as in the units model.

How the parts learn
-------------------

A contest's reading -- one offense's blended success rate and EPA per play
against one defense -- is expected at

    y = center + k (edge + L_off - L_def) + O_off - D_def,    k = c / 2S

so a team's own offense reading and its opponent's both say something about
its level, and each says something about one side. Each team is stepped on
the two readings jointly, (L, O, D) with the opponent held at its mean and its
deviation folded into the noise, the way the record folds in an opponent's:
a three-state Kalman step on two observations, the cross terms dropped after.
What the margin said beyond the readings, `m - S (y_h - y_a)`, steps R alone.

A game with no plays steps L alone on the whole margin, at `obs_sd`. Every
game below FBS that ESPN has no play-by-play for is one, so for those teams L
is the record and nothing else moves.

The season
----------

Over the summer every part folds into L at its own carry -- `offense_carry`,
`defense_carry`, `rest_carry` -- and O, D and R start again from 0 at their
initial deviations. What the summer knows lands on the part it is about:
returning offense on O, returning defense on D, a new quarterback on O after
his first game with plays, a coach who left and the change in talent on L.
The talent edge, home field, rest and a missing quarterback are the parent's
matchup terms, in the edge.

The smoother re-walks the season the way the parent's does: every part
re-stepped from preseason, the opponent at the mean the last pass ended
with; only the means come out of it.
"""

import math
from typing import Any, NamedTuple

from endgame.types import Game

from cassandra.predictor.distribution import NormalMargin, collapse
from cassandra.predictor.glicko import GlickoPredictor, _Rating
from cassandra.predictor.types import Matchup, Prediction
from cassandra.predictor.unit_glicko import (
    LEFT_FOR_JOB,
    UnitMarginGlickoPredictor,
    _Sides,
    _widened,
)


class _Game(NamedTuple):
    """One game as the forward pass saw it, for the smoother."""

    home: str
    away: str
    edge: float  # rating units, toward home
    margin: float
    read: bool
    y_home: float
    y_away: float
    noise_home: float
    noise_away: float
    center: float


def read_step(
    level: _Rating,
    offense: _Rating,
    defense: _Rating,
    opp_level: float,
    opp_level_sd: float,
    opp_offense: float,
    opp_offense_sd: float,
    opp_defense: float,
    opp_defense_sd: float,
    y_own: float,
    y_opp: float,
    noise_own: float,
    noise_opp: float,
    center: float,
    edge: float,
    k: float,
) -> tuple[_Rating, _Rating, _Rating]:
    """One team's (L, O, D) stepped on its game's two readings.

    `edge` is from this team's side. The opponent's level enters both
    readings, with opposite signs, so the two observations' noise is
    correlated through it.
    """
    a = level.rating_deviation**2
    p = offense.rating_deviation**2
    q = defense.rating_deviation**2
    a_opp = opp_level_sd**2
    v1 = y_own - (center + k * (edge + level.rating - opp_level) + offense.rating - opp_defense)
    v2 = y_opp - (center + k * (-edge + opp_level - level.rating) + opp_offense - defense.rating)
    ka = k * k * a
    ka_opp = k * k * a_opp
    s11 = ka + p + noise_own + opp_defense_sd**2 + ka_opp
    s22 = ka + q + noise_opp + opp_offense_sd**2 + ka_opp
    s12 = -ka - ka_opp
    det = s11 * s22 - s12 * s12
    i11, i22, i12 = s22 / det, s11 / det, -s12 / det
    w1 = i11 * v1 + i12 * v2
    w2 = i12 * v1 + i22 * v2
    return (
        _Rating(
            level.rating + k * a * (w1 - w2),
            math.sqrt(max(a - ka * a * (i11 - 2 * i12 + i22), 1e-12)),
        ),
        _Rating(offense.rating + p * w1, math.sqrt(max(p - p * p * i11, 1e-12))),
        _Rating(defense.rating - q * w2, math.sqrt(max(q - q * q * i22, 1e-12))),
    )


def scalar_step(own: _Rating, opp_sd: float, innovation: float, noise: float, scale: float) -> _Rating:
    """A one-state Kalman step on `innovation`, in units of `scale` per state unit."""
    v = own.rating_deviation**2
    total = v + (noise + (scale * opp_sd) ** 2) / scale**2
    gain = v / total
    return _Rating(own.rating + gain * innovation / scale, math.sqrt(v * (1 - gain)))


class ComponentMarginPredictor(UnitMarginGlickoPredictor):
    """Level, offense, defense and rest per team, summed into a Normal margin.

    Built on the units model for its plumbing -- anchors, matchup terms, the
    reading, the center, the summer's facts -- and replacing everything that
    combines or steps. `unit_weight` and `prediction_scale` are not read.
    """

    def __init__(
        self,
        league: str,
        *,
        offense_carry: float = 1.0,
        defense_carry: float = 1.0,
        rest_carry: float = 0.0,
        rest_initial_sd: float = 2.0,
        rest_weekly_sd_increase: float = 0.0,
        rest_obs_sd: float = 12.0,
        game_sd: float = 16.0,
        uncertainty_scale: float = 1.0,
        use_readings: float = 1.0,
        returning_on_level: float = 0.0,
        persist_sides: float = 0.0,
        level_from_margin: float = 0.0,
        earned_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(league, **kwargs)
        self._carry = (offense_carry, defense_carry, rest_carry)
        self._rest_initial_sd = rest_initial_sd
        self._rest_weekly = rest_weekly_sd_increase
        self._rest_obs_sd = rest_obs_sd
        self._game_sd = game_sd
        self._uncertainty_scale = uncertainty_scale
        # Diagnostics: 0 steps every game as one without plays (a record and
        # nothing else); 1 puts returning production on L, as the units model does.
        self._use_readings = bool(use_readings)
        self._returning_on_level = bool(returning_on_level)
        # 1 keeps each side over the summer at its carry, widened by its own
        # season increase, instead of folding it into L: a side's own memory.
        self._persist_sides = bool(persist_sides)
        # 1 steps L on the whole margin in every game, the record's way, and
        # leaves the readings to O and D alone (deviations from what L implies).
        self._level_from_margin = bool(level_from_margin)
        # Above 0, a side counts in the prediction only by what it has earned
        # past its prior, the units model's shrink kept inside the sum: a
        # side's mean times w e / (w e + 1 / prior), e its precision less the
        # prior's. 1 is 1 - var / prior. 0 is every side at full weight.
        self._earned_weight = earned_weight
        self._rest: dict[str, _Rating] = {}
        self._rest_preseason: dict[str, _Rating] = {}
        self._game_weeks: list[list[_Game]] = []
        self._week_games: list[_Game] = []
        #: game id -> (mu, sd) of the prediction made before the game.
        self.recorded: dict[str, tuple[float, float]] = {}
        #: game id -> (expected reading gap, read gap, level gap in points) per read game.
        self.contests: dict[str, tuple[float, float, float]] = {}

    # ---- the parts -------------------------------------------------------

    def _rest_of(self, team: str) -> _Rating:
        return self._rest.get(team, _Rating(0.0, self._rest_initial_sd))

    @property
    def _per_play(self) -> float:
        return self.points_per_rating / (2 * self._points_per_epa)

    def predict_distribution(self, matchup: Matchup) -> NormalMargin:
        c, s = self.points_per_rating, self._points_per_epa
        home, away = self.get_rating(matchup.home), self.get_rating(matchup.away)
        sh, sa = self._offsets(matchup.home), self._offsets(matchup.away)
        edge = self.home_edge(matchup) + self.matchup_adjustment(matchup)
        mu = (
            c * (home.rating - away.rating + edge)
            + s * (self._counted(sh) - self._counted(sa))
            + self._rest_of(matchup.home).rating
            - self._rest_of(matchup.away).rating
        )
        var = self._game_sd**2 + self._uncertainty_scale * c * c * (
            home.rating_deviation**2 + away.rating_deviation**2
        )
        return NormalMargin(mu, math.sqrt(var), "points")

    def _counted(self, pair: _Sides) -> float:
        """O + D as the prediction counts them; see `earned_weight`."""
        if not self._earned_weight:
            return pair.offense.rating + pair.defense.rating
        total = 0.0
        for unit, initial in ((pair.offense, self._initial_sd.offense), (pair.defense, self._initial_sd.defense)):
            prior = 1 / initial**2
            earned = self._earned_weight * max(0.0, 1 / unit.rating_deviation**2 - prior)
            total += unit.rating * earned / (earned + prior)
        return total

    # ---- one game --------------------------------------------------------

    def _step_pair(
        self,
        g: _Game,
        levels: dict[str, _Rating],
        sides: dict[str, _Sides],
        rest: dict[str, _Rating],
        held: tuple[dict, dict, dict] | None,
    ) -> dict[str, tuple[_Rating, _Sides, _Rating]]:
        """Both teams' new parts from their pre-game ones.

        `held` is the smoother's settled means to read an opponent at; None
        in the forward pass, where the opponent is read as it stands.
        """
        c, s, k = self.points_per_rating, self._points_per_epa, self._per_play

        def state(team: str):
            level = levels.get(team)
            if level is None:
                level = self.get_rating(team) if held is None else _Rating(
                    self.anchor(team), self._initial_rd)
            pair = sides.get(team, _Sides(
                _Rating(0.0, self._initial_sd.offense), _Rating(0.0, self._initial_sd.defense)))
            return level, pair, rest.get(team, _Rating(0.0, self._rest_initial_sd))

        before = {g.home: state(g.home), g.away: state(g.away)}

        def opp_means(team: str):
            level, pair, r = before[team]
            if held is None:
                return level.rating, pair.offense.rating, pair.defense.rating, r.rating
            h_levels, h_sides, h_rest = held
            hl = h_levels.get(team, level)
            hp = h_sides.get(team, pair)
            hr = h_rest.get(team, r)
            return hl.rating, hp.offense.rating, hp.defense.rating, hr.rating

        out = {}
        for me, opp, sign in ((g.home, g.away, 1.0), (g.away, g.home, -1.0)):
            level, pair, r = before[me]
            o_level, o_pair, o_r = before[opp]
            ol, oo, od, orr = opp_means(opp)
            if g.read:
                y_own, y_opp = (g.y_home, g.y_away) if sign > 0 else (g.y_away, g.y_home)
                n_own, n_opp = (g.noise_home, g.noise_away) if sign > 0 else (g.noise_away, g.noise_home)
                new_level, new_off, new_def = read_step(
                    level._replace(rating_deviation=1e-6) if self._level_from_margin else level,
                    pair.offense, pair.defense,
                    ol, o_level.rating_deviation,
                    oo, o_pair.offense.rating_deviation,
                    od, o_pair.defense.rating_deviation,
                    y_own, y_opp, n_own, n_opp, g.center, sign * g.edge, k,
                )
                residual = sign * (g.margin - s * (g.y_home - g.y_away))
                new_r = scalar_step(r, o_r.rating_deviation, residual - (r.rating - orr),
                                    self._rest_obs_sd**2, 1.0)
                if self._level_from_margin:
                    expected = (
                        c * (level.rating + sign * g.edge - ol)
                        + s * (pair.offense.rating + pair.defense.rating - oo - od)
                        + r.rating - orr
                    )
                    new_level = scalar_step(level, o_level.rating_deviation,
                                            sign * g.margin - expected, self._obs_sd**2, c)
                out[me] = (new_level, _Sides(new_off, new_def), new_r)
            else:
                expected = (
                    c * (level.rating + sign * g.edge - ol)
                    + s * (pair.offense.rating + pair.defense.rating - oo - od)
                    + r.rating - orr
                )
                new_level = scalar_step(level, o_level.rating_deviation,
                                        sign * g.margin - expected, self._obs_sd**2, c)
                out[me] = (new_level, pair, r)
        return out

    def _update_game(self, game: Game) -> Prediction:
        distribution = self.predict_distribution(game)
        self.recorded[game.game_id] = (distribution.mean, distribution.sd)
        edge = self.home_edge(game) + self.matchup_adjustment(game)
        epa = self._game_epa.get(game.game_id)
        read = self._use_readings and epa is not None and bool(epa.home_plays) and bool(epa.away_plays)
        if read:
            y_home, y_away = self._reading(epa, self._success_weight)
            noise = (self._play_sd**2 / epa.home_plays, self._play_sd**2 / epa.away_plays)
        else:
            y_home = y_away = 0.0
            noise = (0.0, 0.0)
        if read:
            sh, sa = self._offsets(game.home), self._offsets(game.away)
            level_gap = self.get_rating(game.home).rating - self.get_rating(game.away).rating
            self.contests[game.game_id] = (
                2 * self._per_play * (edge + level_gap)
                + sh.offense.rating + sh.defense.rating - sa.offense.rating - sa.defense.rating,
                y_home - y_away,
                self.points_per_rating * (edge + level_gap),
            )
        g = _Game(game.home, game.away, edge, float(game.home_score - game.away_score),
                  read, y_home, y_away, noise[0], noise[1], self.epa_center)
        stepped = self._step_pair(g, self._ratings, self._sides, self._rest, None)
        for team, (level, pair, r) in stepped.items():
            self._ratings[team] = level
            if read:
                self._sides[team] = pair
                self._rest[team] = r
        if read:
            self._center_sum += y_home + y_away
            self._center_count += 2
            for team in (game.home, game.away):
                if team in self._qb_pending:
                    self._qb_pending.discard(team)
                    self._shift_side(team, "offense", self._quarterback_points(team), preseason=True)
        self._week_games.append(g)
        self._prior_manager.add_game(game)
        self._adjustments.record(game)
        return collapse(distribution)

    # ---- the week and the smoother ---------------------------------------

    def pass_week(self) -> None:
        self._game_weeks.append(self._week_games)
        self._week_games = []
        self._ratings = self._aged(self._ratings, self._weekly_rd_increase)
        self._sides = self._aged_sides(self._sides, self._weekly_sd_increase)
        self._rest = self._aged_rest(self._rest)
        for _ in range(self._passes - 1):
            self._smooth_parts()

    def _aged_rest(self, rest: dict[str, _Rating]) -> dict[str, _Rating]:
        if not self._rest_weekly:
            return rest
        return {t: _widened(r, self._rest_weekly, self._rest_initial_sd) for t, r in rest.items()}

    def _smooth_parts(self) -> None:
        played = {t for week in self._game_weeks for g in week for t in (g.home, g.away)}
        if not played:
            return
        levels = {t: self._preseason[t] for t in played if t in self._preseason}
        sides = {t: self._sides_preseason[t] for t in played if t in self._sides_preseason}
        rest = {t: self._rest_preseason[t] for t in played if t in self._rest_preseason}
        held = (self._ratings, self._sides, self._rest)
        for week in self._game_weeks:
            for g in week:
                stepped = self._step_pair(g, levels, sides, rest, held)
                for team, (level, pair, r) in stepped.items():
                    levels[team] = level
                    if g.read:
                        sides[team] = pair
                        rest[team] = r
            levels = self._aged(levels, self._weekly_rd_increase)
            sides = self._aged_sides(sides, self._weekly_sd_increase)
            rest = self._aged_rest(rest)
        for team in played:
            if team in levels and team in self._ratings:
                self._ratings[team] = self._ratings[team]._replace(rating=levels[team].rating)
            if team in sides and team in self._sides:
                now = self._sides[team]
                self._sides[team] = _Sides(
                    now.offense._replace(rating=sides[team].offense.rating),
                    now.defense._replace(rating=sides[team].defense.rating),
                )
            if team in rest and team in self._rest:
                self._rest[team] = self._rest[team]._replace(rating=rest[team].rating)

    # ---- the summer ------------------------------------------------------

    def _shift_side(self, team: str, side: str, points: float, preseason: bool) -> None:
        if not points:
            return
        by = points / self._points_per_epa
        books = (self._sides, self._sides_preseason) if preseason else (self._sides,)
        for book in books:
            pair = book.get(team, _Sides(
                _Rating(0.0, self._initial_sd.offense), _Rating(0.0, self._initial_sd.defense)))
            unit = getattr(pair, side)
            book[team] = pair._replace(**{side: unit._replace(rating=unit.rating + by)})

    def _roll_over(self) -> None:
        c, s = self.points_per_rating, self._points_per_epa
        offense_carry, defense_carry, rest_carry = self._carry
        kept_sides = {}
        if self._persist_sides:
            for team, pair in self._sides.items():
                kept_sides[team] = _Sides(
                    _widened(pair.offense._replace(rating=offense_carry * pair.offense.rating),
                             self._season_sd_increase.offense, self._initial_sd.offense),
                    _widened(pair.defense._replace(rating=defense_carry * pair.defense.rating),
                             self._season_sd_increase.defense, self._initial_sd.defense),
                )
        for team in set(self._sides) | set(self._rest):
            points = 0.0
            pair = self._sides.get(team)
            if pair is not None and not self._persist_sides:
                points += s * (offense_carry * pair.offense.rating + defense_carry * pair.defense.rating)
            r = self._rest.get(team)
            if r is not None:
                points += rest_carry * r.rating
            if points and team in self._ratings:
                level = self._ratings[team]
                self._ratings[team] = level._replace(rating=level.rating + points / c)
        GlickoPredictor._roll_over(self)
        if self._center_count:
            self._center_previous = self._center_sum / self._center_count
        self._center_sum, self._center_count = 0.0, 0
        self._sides, self._rest = kept_sides, {}
        self._game_weeks, self._week_games = [], []
        self._qb_pending = set()
        for team, fact in self._offseason.seasons(self._season):
            if fact.coach_departure == LEFT_FOR_JOB:
                self._shift_rating(team, self._coach_left_shift)
            if fact.new_quarterback and (self._new_qb_shift or self._qb_quality_shift):
                self._qb_pending.add(team)
        if self._season is not None:
            if self._talent_shift:
                for team, level in self._talent.season(self._season):
                    change = level - self._talent.previous(team, self._season)
                    self._shift_rating(team, self._talent_shift * change)
            for weight, index, side in (
                (self._offense_returning_shift, self._offense_returning, "offense"),
                (self._defense_returning_shift, self._defense_returning, "defense"),
            ):
                if weight:
                    for team, level in index.season(self._season):
                        if self._returning_on_level:
                            self._shift_rating(team, weight * level)
                        else:
                            self._shift_side(team, side, weight * level, preseason=False)
        self._sides_preseason = dict(self._sides)
        self._rest_preseason = dict(self._rest)

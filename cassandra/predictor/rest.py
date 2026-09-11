"""Days off, and what a rest gap between two teams is worth.

A team coming off a bye plays a team that played six days ago, and the model
prices them as if both were fresh. The residuals say that costs something:
on ncaafb, FBS against FBS, the home team beats its expected margin by 1.26
points when it has 4-7 days more rest and by 2.43 with 8 or more, against
0.05 in the games where both sides are level.

That is a small axis -- 0.43 points of signal against a margin MAE of 12.85,
and 2.35 sigma, under the 3 `diagnose.py` calls flat -- and it is built
anyway for one reason the bigger axes don't have: it costs no new data. Every
date it needs is already in the `Game` the replay walks, so this is a
parameter over information the model was already holding and throwing away.
Whether it earns its place is then a question the search answers, and 0
turns it off.

Unlike a home advantage this applies at a neutral site too. Nobody is at
home in a bowl game and both teams still arrived on different amounts of
rest.

## What a consumer of a release gets

Nothing, for now, and that is worth saying out loud. `ModelRelease` carries
ratings and parameters; it does not carry when each team last played, so a
`rating_predictor()` rebuilt from one starts with an empty ledger and prices
every matchup at zero rest differential until it has walked some games. That
is the safe direction -- a missing date reads as "no information" rather
than as a wrong number -- but it means a live prediction off a release will
not use this feature while a replayed one does. Carrying a last-played date
per team in the release is what would close that, and it is deliberately not
done here: the parameter has to earn its place in a search first.
"""

from datetime import datetime

from endgame.types import Game

from .types import Matchup

#: Rating points for arriving off a bye when the other side isn't. A flat
#: bump, not a rate: see `RestLedger` for why this is a threshold rather than
#: a slope. 0 is the feature switched off, which is what every model
#: published before this replayed with.
DEFAULT_REST_ADVANTAGE = 0.0

#: How much longer one side's gap has to be before it counts as a bye.
#:
#: 5 days, because college football is played on Saturdays: a team on its
#: normal week has a 7-day gap, one coming off a bye has 14, and the
#: interesting cases are the ones that clear a week. Below this the
#: differential is a Thursday game or a Friday game, which is a different
#: fact and, measured, not a consistent one -- the +2 to +3 day bucket runs
#: the *opposite* way to the +4 to +7 bucket.
REST_THRESHOLD_DAYS = 5.0

#: Past this, a gap stops being evidence of rest and starts being evidence
#: about the data. A team is not off for three weeks in the middle of a
#: football season; far more often the game it played is one the season file
#: doesn't have, and reading the hole as a bye invents the largest rest
#: advantage of that team's year out of a missing row.
#:
#: 20 days: a real bye is 14 and two in a row is 21, so this sits above every
#: ordinary schedule and below the gaps that are usually absences. It does
#: also catch the genuine long layoff -- a conference championship to a bowl
#: is about four weeks -- and that is the right trade, because those are
#: games where *both* sides have been off for a month and the differential is
#: near zero anyway.
#:
#: Measured on ncaafb, 1.6% of team-games have a gap this long, and they
#: account for 10% of the games the bye term would otherwise fire on.
REST_MAX_GAP_DAYS = 20.0


def validated_rest_advantage(value: float) -> float:
    """Check a `rest_advantage` on its way into a predictor.

    Negative is refused rather than searched: it would mean a rested team is
    worse for being rested. 0 is the off switch and stays legal.
    """
    if value < 0:
        raise ValueError(f"rest_advantage must be non-negative, got {value}")
    return value


class RestLedger:
    """When each team last played, and whether one side is coming off a bye.

    **A threshold, not a slope.** The first version of this priced rating
    points per day of differential, and that form fights the data: the
    per-bucket residual on ncaafb is not monotone in days, with the +2 to +3
    day bucket running -0.94 while +4 to +7 runs +1.26, so a line drawn
    through them is wrong in the middle by construction. Measured against the
    residuals directly, the best linear-in-days correction recovers 0.00108
    of margin MAE and the best bucketed one 0.00217 -- the shape is worth
    about as much as the effect.

    So the question this asks is the one worth asking: did one side come off
    a bye and the other not? Anything past the threshold is the same answer,
    which also disposes of the long-layoff problem the old day cap existed
    for -- a team whose last game was a bowl is simply "rested", not
    thirty-seven days of rested.

    Held by `Predictor` rather than by a mixin, the way `_anchors` and
    `_season_regression` are: the base keeps an inert one so every predictor
    can ask for the adjustment, and the subclasses that expose the parameter
    replace it with a live one.

    The ledger is state, not configuration -- it is rebuilt by walking games
    and is reset at a season boundary, since a team's last game of one season
    says nothing about how rested it is for the next.
    """

    def __init__(
        self,
        points: float = DEFAULT_REST_ADVANTAGE,
        threshold_days: float = REST_THRESHOLD_DAYS,
        max_gap_days: float = REST_MAX_GAP_DAYS,
    ) -> None:
        self._points = validated_rest_advantage(points)
        self._threshold_days = threshold_days
        self._max_gap_days = max_gap_days
        self._last_played: dict[str, datetime] = {}

    @property
    def points(self) -> float:
        """Rating points for the rested side. 0 means switched off."""
        return self._points

    @property
    def threshold_days(self) -> float:
        """How much longer a gap has to be before it counts as a bye."""
        return self._threshold_days

    @property
    def max_gap_days(self) -> float:
        """Past which a gap is read as missing data rather than as rest."""
        return self._max_gap_days

    def record(self, game: Game) -> None:
        """Note that both sides played on this date.

        Called from `update_game` after the prediction is made, so a game
        never contributes to its own rest calculation.
        """
        self._last_played[game.home] = game.date
        self._last_played[game.away] = game.date

    def reset(self) -> None:
        """Forget every date. Called at a season boundary."""
        self._last_played.clear()

    def _days_off(self, team: str, date: datetime) -> float | None:
        """Days since this team last played, or None if it hasn't this season.

        None rather than a large number for a season opener: "we have no
        idea" and "extremely rested" are different claims, and only one of
        them is true in week one.
        """
        last = self._last_played.get(team)
        if last is None:
            return None
        # Season pickles carry naive datetimes in some leagues and aware ones
        # in others. Both sides of this subtraction come out of the same
        # league's games, so they always agree with each other.
        return (date - last).total_seconds() / 86400.0

    def differential(self, matchup: Matchup) -> float:
        """Home days off minus away days off, 0 when either is unknown.

        0 for an opener is the honest answer and the conservative one: with
        one side's rest unknown there is no differential to price, and
        guessing would put the largest adjustment of the season on the game
        the model knows least about.

        Unclamped -- `rested_side` is what turns this into an adjustment, and
        a caller reading it for a diagnostic wants the real number of days.
        """
        home = self._days_off(matchup.home, matchup.date)
        away = self._days_off(matchup.away, matchup.date)
        if home is None or away is None:
            return 0.0
        return home - away

    def rested_side(self, matchup: Matchup) -> float:
        """+1 if the home side came off the longer break, -1 if the away side.

        0 when neither cleared the threshold, which is most games: college
        football is Saturday to Saturday, so the usual differential is zero.

        Also 0 when *either* side's gap is `max_gap_days` or longer, whatever
        the differential says. A three-week hole in a football season is more
        often a game the data is missing than a break the team took, and
        reading it as a bye would put the biggest adjustment of that team's
        season on the row that isn't there. One suspect side is enough to
        throw the comparison out: the differential is a difference, and it is
        only as trustworthy as its worse half.
        """
        home = self._days_off(matchup.home, matchup.date)
        away = self._days_off(matchup.away, matchup.date)
        if home is None or away is None:
            return 0.0
        if home >= self._max_gap_days or away >= self._max_gap_days:
            return 0.0
        difference = self.differential(matchup)
        if difference >= self._threshold_days:
            return 1.0
        if difference <= -self._threshold_days:
            return -1.0
        return 0.0

    def adjustment(self, matchup: Matchup) -> float:
        """Rating points to add to the home side for the rest gap.

        Applied at a neutral site as well, unlike a home advantage: nobody is
        at home in a bowl and both teams still arrived on different rest.
        """
        if not self._points:
            return 0.0
        return self._points * self.rested_side(matchup)

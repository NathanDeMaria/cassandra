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

#: Rating points per day of rest advantage. 0 is the feature switched off,
#: which is what every model published before this replayed with, so a
#: release that names no value replays exactly as it did.
DEFAULT_REST_ADVANTAGE = 0.0

#: How many days of differential the model is willing to read. Past this the
#: gap stops being rest and starts being a different kind of fact: a team
#: whose last game was a bowl and one that finished in November differ by
#: five weeks, and no amount of that is recovery. Also what keeps the first
#: game after a mid-season cancellation from dominating a team's season.
#:
#: 14 because two weeks covers a bye plus its neighbouring week, which is the
#: largest gap that happens on purpose in a football season.
REST_CAP_DAYS = 14.0


def validated_rest_advantage(value: float) -> float:
    """Check a `rest_advantage` on its way into a predictor.

    Negative is refused rather than searched. It would mean a rested team is
    worse for being rested, and a search that wandered there would be fitting
    the handful of long-gap games -- bowls, cancellations -- that
    `REST_CAP_DAYS` exists to bound in the first place. 0 is the off switch
    and stays legal.
    """
    if value < 0:
        raise ValueError(f"rest_advantage must be non-negative, got {value}")
    return value


class RestLedger:
    """When each team last played, and what today's gap is worth.

    Held by `Predictor` rather than by a mixin, the way `_anchors` and
    `_season_regression` are: the base keeps an inert one so every predictor
    can ask for the adjustment, and the subclasses that expose the parameter
    replace it with a live one.

    The ledger is state, not configuration -- it is rebuilt by walking games
    and is reset at a season boundary, since a team's last game of one season
    says nothing about how rested it is for the next.
    """

    def __init__(
        self, per_day: float = DEFAULT_REST_ADVANTAGE, cap_days: float = REST_CAP_DAYS
    ) -> None:
        self._per_day = validated_rest_advantage(per_day)
        self._cap_days = cap_days
        self._last_played: dict[str, datetime] = {}

    @property
    def per_day(self) -> float:
        """Rating points per day of differential. 0 means switched off."""
        return self._per_day

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
        """Home days off minus away days off, clamped, 0 when either is unknown.

        0 for an opener is the honest answer and the conservative one: with
        one side's rest unknown there is no differential to price, and
        guessing would put the largest adjustment of the season on the game
        the model knows least about.
        """
        home = self._days_off(matchup.home, matchup.date)
        away = self._days_off(matchup.away, matchup.date)
        if home is None or away is None:
            return 0.0
        return max(-self._cap_days, min(self._cap_days, home - away))

    def adjustment(self, matchup: Matchup) -> float:
        """Rating points to add to the home side for the rest gap.

        Applied at a neutral site as well, unlike a home advantage: nobody is
        at home in a bowl and both teams still arrived on different rest.
        """
        if not self._per_day:
            return 0.0
        return self._per_day * self.differential(matchup)

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

No dates. `ModelRelease` carries ratings and parameters; it does not carry
when each team last played, so a `rating_predictor()` rebuilt from one starts
with an empty `RestLedger` and prices every matchup at zero rest differential
until it has walked some games. That is the safe direction -- a missing date
reads as "no information" rather than as a wrong number -- and it means a
live prediction off a release does not use this feature while a replayed one
does.

What closes that is the seam `qb_out` already has. "Did one side come off the
longer break?" is asked of a `RestSource`, and a caller who knows the answer
hands one over instead of the walking ledger:

    sources = MatchupSources.for_league(league)._replace(
        rest=StatedRest({"401752708": "LSU Tigers"})
    )

which is what rest looks like when it comes from a schedule somebody read
rather than from games a replay walked. Carrying a last-played date per team
*in the release* would close it a second way and is still deliberately not
done: that changes the artifact format, and this does not.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

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


class RestSource(Protocol):
    """Where the answer to "who is rested?" comes from.

    Two implementations, and the difference between them is where the fact
    was learned rather than what it means: `RestLedger` derives it from the
    dates of games it has walked, and `StatedRest` is handed it. A predictor
    asks the same question of either, which is what keeps `predict_game` the
    one interface -- a what-if about a fixture and a replay of 2014 differ in
    what they hand the predictor, not in how they call it.

    `record` and `reset` are on the protocol rather than on the ledger alone
    because the replay calls them on whatever it has. A source that was told
    its answer has nothing to learn from a game and nothing to forget at a
    season boundary, so it implements both as no-ops.
    """

    def rested_side(self, matchup: Matchup) -> float:
        """+1 if the home side came off the longer break, -1 if the away side."""
        ...

    def record(self, game: Game) -> None:
        """Note a played game."""
        ...

    def reset(self) -> None:
        """Cross a season boundary."""
        ...


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

    Reached through `MatchupSources`, which is what a predictor is handed and
    what a caller swaps to state the answer instead. What it is *worth* is not
    here: `rest_advantage` is a weight, it is fit by a search and rides in a
    release's parameters, and keeping it on `MatchupAdjustments` beside
    `qb_out_penalty` is what makes the two terms the same shape.

    The ledger is state, not configuration -- it is rebuilt by walking games
    and is reset at a season boundary, since a team's last game of one season
    says nothing about how rested it is for the next.
    """

    def __init__(
        self,
        threshold_days: float = REST_THRESHOLD_DAYS,
        max_gap_days: float = REST_MAX_GAP_DAYS,
    ) -> None:
        self._threshold_days = threshold_days
        self._max_gap_days = max_gap_days
        self._last_played: dict[str, datetime] = {}

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

class StatedRest:
    """Who came off the longer break, by game, because somebody said so.

    The counterpart to `RestLedger` for the case it cannot serve: a fixture,
    where the dates a ledger would need are in a schedule the replay has not
    walked. Built in memory from what a caller knows, the way `QbOutIndex` is,
    and keyed the same way -- game id to the team that is rested.

    A team rather than a sign, for the reason the quarterback index holds
    names: a caller states a fact it can check ("Michigan is off a bye"), and
    which direction that pushes the number is this module's business. A sign
    at the call site is a sign convention at the call site, and that is the
    kind of thing that gets inverted once and noticed a season later.

    Nobody is rested in a game the mapping does not mention, which is the same
    answer a ledger gives when neither side clears the threshold and the same
    one `QbOutIndex` gives for a game it has never seen.
    """

    def __init__(self, games: Mapping[str, str] | None = None) -> None:
        self._games = dict(games or {})

    def __len__(self) -> int:
        return len(self._games)

    def rested_side(self, matchup: Matchup) -> float:
        """+1 if the home side is the stated one, -1 if the away side.

        0 for a game nobody stated, and 0 for a stated team that is not in
        this one: a name that matches neither side is a typo or a stale
        fixture, and reading it as "the other team is rested" would turn a bad
        input into the largest adjustment on the board.
        """
        rested = self._games.get(matchup.game_id)
        if rested is None:
            return 0.0
        if rested == matchup.home:
            return 1.0
        if rested == matchup.away:
            return -1.0
        return 0.0

    def record(self, game: Game) -> None:
        """Nothing: a stated fact does not move as games are walked."""

    def reset(self) -> None:
        """Nothing: there are no dates here for a season boundary to void."""

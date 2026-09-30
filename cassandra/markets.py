"""The prediction markets as a book: gold-rush's prices in `OddsDatabase` form.

gold-rush writes Kalshi's and Polymarket's hourly prices on every game it
could match to ESPN, next to the odds pulls in the same bucket:

    markets/{venue}/{league}/{YYYY-MM-DD}.json

keyed by the same ESPN `game_id` the odds are, so nothing here maps one id
onto another. See gold-rush's README for the file; the parts read here are
each game's `home` and `away` sides, each a list of rows in the file's
`fields` order -- `at` (epoch seconds, the end of an hour), then what that
side's contract cost on a $1 payout.

A contract's cost *is* an implied probability, and a moneyline is just
another way of writing one, so each hour becomes an `OddsSnapshot` whose
moneylines are the American price of buying each side then. That is the
whole trick: `cassandra.betting`'s moneyline calibration, strategies and
line windows run on these unchanged. There is no spread -- the venues trade
who wins -- so `spread` is always None and the spread half of a betting
report has nothing to grade.

**Which price.** The one a bettor would pay. Kalshi keeps an order book per
team and the file carries its closing ask, which is what buying costs; its
last trade can be hours stale on a quiet market and is missing outright on
some rows, while the ask was on every pre-kickoff row of the files checked
when this was written. Polymarket
keeps one number an hour and nothing else, so that's its price.

**Fees.** Kalshi charges a taker fee per contract, `KALSHI_TAKER_FEE_RATE`
x p x (1 - p), which is added to the ask: at 50 cents that's 1.75 cents a
contract, a real slice of any edge. Polymarket's international venue has
not charged a trading fee on most markets, so its prices are taken as they
are. Neither was checked against a current fee schedule when this was
written -- they are constants because schedules change, and a moneyline ROI
here is only as right as they are. Kalshi also rounds its fee up to the
cent per order; with no order size to round, that's left out.

The two sides' costs sum to more than a dollar on Kalshi -- the gap between
ask and bid on each book, plus the fee -- the way a sportsbook's two
implied probabilities sum to more than one. `no_vig_home_probability`
normalizes it out the same way for both.
"""

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from endgame_aws.io import get_session, list_keys, read_from_s3

from .odds import OddsDatabase, OddsSnapshot

VENUES = ("kalshi", "polymarket")

# Kalshi's taker fee is this times price times (1 - price), per contract.
KALSHI_TAKER_FEE_RATE = 0.07

# The most a game's two sides may cost together for the hour to count as a
# price. A Kalshi market opens days out with asks near a dollar on both sides
# -- the entry read on 2025-26 men's basketball had a median of 1.61 -- and
# no-vig on a book that wide is about 50/50 whatever the game, which would
# hand the model a phantom edge on every one. Closes run 1.02-1.07. Polymarket
# quotes one number for both sides, which always sums to about 1, so it can't
# be caught this way -- see `ENTRY_VENUES`.
MAX_OVERROUND = 1.10

# The venues whose early prices can be told apart from an empty market, and so
# whose entry read is worth grading. Kalshi's file carries the quotes, and a
# book nobody is making shows up as too wide (`MAX_OVERROUND`). Polymarket's
# carries one number an hour and no quote or volume, and its markets post
# prices days before anyone trades them: on January 2026 men's basketball its
# reads 48-96 hours out sat 0.12 from the close on average, Kalshi's 0.04,
# and graded as entries they "beat" the market by 20%+ ROI. Its close, an
# hour before tip-off, is as good as Kalshi's; its entry is noise.
ENTRY_VENUES = ("kalshi",)

# One day's file; the `_pulls/` summaries beside them don't match.
_DAY_KEY_RE = re.compile(r"markets/[a-z]+/[^/]+/\d{4}-\d{2}-\d{2}\.json$")

# Polymarket stamps a side's hour a few seconds past it, and the two sides of
# one game a few seconds apart; Kalshi's are on the hour. Rows pair by the
# hour they're nearest.
_HOUR = 3600


def probability_to_american(cost: float | None) -> float | None:
    """The American moneyline that costs `cost` per dollar of payout.

    The inverse of `betting.american_to_probability`: 0.60 is -150 (risk 150
    to win 100), 0.25 is +300. None for a price that isn't strictly between
    0 and 1 -- a settled or empty book isn't a price anyone could bet.
    """
    if cost is None or not 0 < cost < 1:
        return None
    if cost >= 0.5:
        return -100 * cost / (1 - cost)
    return 100 * (1 - cost) / cost


def kalshi_fee(price: float) -> float:
    """Kalshi's taker fee on one contract bought at `price`, in dollars."""
    return KALSHI_TAKER_FEE_RATE * price * (1 - price)


def _cost(venue: str, row: Sequence, fields: Sequence[str]) -> float | None:
    """What buying this side cost at this row, fee included."""
    if venue == "kalshi":
        ask = row[fields.index("ask")]
        return None if ask is None else ask + kalshi_fee(ask)
    return row[fields.index("price")]


def game_snapshots(game: dict, venue: str, fields: Sequence[str]) -> list[OddsSnapshot]:
    """One game's hourly prices as snapshots, oldest first.

    An hour one side has and the other doesn't is kept with the missing
    side None, as `odds._parse_snapshot` keeps a game with one moneyline
    off. `read_at` is the later of the two sides' stamps for the hour, the
    first moment both prices were known.

    An hour whose two sides cost more than `MAX_OVERROUND` together isn't a
    market yet and is left out.
    """
    at = fields.index("at")
    hours: dict[int, dict[str, tuple[int, float | None]]] = {}
    for side in ("home", "away"):
        for row in game[side]["prices"]:
            stamp = int(row[at])
            hour = round(stamp / _HOUR)
            hours.setdefault(hour, {})[side] = (stamp, _cost(venue, row, fields))
    snapshots = []
    for hour in sorted(hours):
        sides = hours[hour]
        home = sides.get("home", (0, None))
        away = sides.get("away", (0, None))
        if (
            home[1] is not None
            and away[1] is not None
            and home[1] + away[1] > MAX_OVERROUND
        ):
            continue
        home_ml, away_ml = (
            probability_to_american(home[1]),
            probability_to_american(away[1]),
        )
        if home_ml is None and away_ml is None:
            continue
        snapshots.append(
            OddsSnapshot(
                read_at=datetime.fromtimestamp(max(home[0], away[0]), UTC),
                spread=None,
                home_moneyline=home_ml,
                away_moneyline=away_ml,
            )
        )
    return snapshots


def history_from_days(
    venue: str, days: Iterable[dict]
) -> dict[str, list[OddsSnapshot]]:
    """Per-game series from parsed day files, keyed by ESPN game id.

    A game appears in one day's file only, but a later pull of the same day
    replaces the file rather than adding to it, so a game id seen twice is
    the same game read twice and the last one wins.
    """
    history: dict[str, list[OddsSnapshot]] = {}
    for day in days:
        fields = day["fields"]
        for game in day["games"]:
            history[str(game["game_id"])] = game_snapshots(game, venue, fields)
    return history


async def _read_days(bucket: str, prefix: str) -> list[dict]:
    session = get_session()
    async with session.create_client("s3") as client:
        keys = [
            key
            async for key in list_keys(bucket, prefix, client)
            if _DAY_KEY_RE.search(key)
        ]
        bodies = await asyncio.gather(
            *(read_from_s3(bucket, key, client) for key in keys)
        )
    return [json.loads(body.decode()) for body in bodies]


async def market_database(bucket: str, venue: str, league: str) -> OddsDatabase:
    """One venue's prices on one league's games, as a book `betting` can read.

    `get_odds` is always None on it -- there are no spreads -- so it's for
    `line_windows` and the moneyline reports, not for `join_with_odds`.
    """
    if venue not in VENUES:
        raise ValueError(f"no market venue {venue!r}; there's {', '.join(VENUES)}")
    days = await _read_days(bucket, f"markets/{venue}/{league}/")
    return OddsDatabase.from_history(history_from_days(venue, days))

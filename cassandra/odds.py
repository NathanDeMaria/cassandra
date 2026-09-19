import asyncio
import json
import re
from datetime import UTC, datetime
from typing import NamedTuple
from zoneinfo import ZoneInfo

from endgame_aws.io import get_session, list_keys, read_from_s3
from pydantic import BaseModel


class Odds(BaseModel):
    game_id: str
    spread: float


class OddsSnapshot(NamedTuple):
    """One book's prices on one game, as they stood at one read.

    `read_at` is when EndGame's odds job read them, not when the game is
    played -- a game accumulates one of these per pull for as long as it's
    on the board, and the sequence is what says how a line moved. The
    spread is quoted from the home side (negative means the home team is
    favored), the same convention as `Odds.spread`; the moneylines are
    American prices. Any of the three is None when the book had it off.
    """

    read_at: datetime
    spread: float | None
    home_moneyline: float | None
    away_moneyline: float | None


# The odds job writes one object per pull, named for when it read:
# odds/<league>/<YYYY-MM-DD>/<HH-MM>[-<horizon>].json, in America/Chicago.
# The horizon suffix arrived when pulls started reaching past the day
# they ran, and the earlier keys don't have it.
_ODDS_KEY_RE = re.compile(
    r"odds/[^/]+/(\d{4}-\d{2}-\d{2})/(\d{2})-(\d{2})(?:-[a-z]+)?\.json$"
)
_ODDS_KEY_TZ = ZoneInfo("America/Chicago")


def read_at_from_key(key: str) -> datetime | None:
    """When the pull that wrote this object read the book, in UTC.

    None for a key that isn't a pull -- nothing else lives under `odds/`
    today, but a stray object there should be skipped, not raise.
    """
    match = _ODDS_KEY_RE.search(key)
    if match is None:
        return None
    day, hour, minute = match.groups()
    local = datetime.fromisoformat(f"{day}T{hour}:{minute}").replace(
        tzinfo=_ODDS_KEY_TZ
    )
    return local.astimezone(UTC)


def _american(price: object) -> float | None:
    """A moneyline as ESPN quotes it -- "-310", "+250", or "OFF" -- as a number."""
    try:
        return float(str(price).replace("+", ""))
    except ValueError:
        return None


def _moneylines(entry: dict) -> tuple[float | None, float | None]:
    """The home and away moneylines from one book's entry.

    ESPN nests these under `moneyline.<side>.close.odds`; `close` is the
    price at the time of the read, `open` is the book's opener. Either can
    read "OFF", which a book does on a game it won't take a moneyline on --
    on ncaafb that's roughly every game past a 30-point spread.
    """
    prices = entry.get("moneyline") or {}

    def side(name: str) -> float | None:
        return _american((((prices.get(name) or {}).get("close")) or {}).get("odds"))

    return side("home"), side("away")


def _parse_odds(game: dict) -> Odds | None:
    """The spread a book had on this game, or None if it never posted one.

    A book that takes the line off the board still writes an entry -- with an
    over/under, a moneyline, and a `pointSpread` whose line reads "OFF" -- but
    with no `spread` key at all. That's a game with no line, which is the same
    thing as a game the odds database doesn't cover: `get_odds` already
    returns None for those, `_build_prediction` records a null spread, and
    `score_predictions` leaves them out of the betting metrics.

    Returning None rather than raising because these are a handful of games in
    tens of thousands, and `read_all_odds` reads every league's odds no matter
    which league is being run -- so one spreadless game used to take down any
    run of any model.
    """
    entries = game.get("odds") or []
    if not entries or "spread" not in entries[0]:
        return None
    return Odds(
        game_id=game["competition_id"],
        spread=entries[0]["spread"],
    )


def _parse_snapshot(game: dict, read_at: datetime) -> OddsSnapshot | None:
    """Everything `_parse_odds` reads plus the moneylines, stamped with the read.

    Unlike `_parse_odds` this keeps a game whose spread is off but whose
    moneyline isn't (or the reverse): a snapshot with one price on it still
    says what the book was doing at that moment. Only a game with no entry
    at all is nothing.
    """
    entries = game.get("odds") or []
    if not entries:
        return None
    home, away = _moneylines(entries[0])
    return OddsSnapshot(
        read_at=read_at,
        spread=entries[0].get("spread"),
        home_moneyline=home,
        away_moneyline=away,
    )


async def _read_snapshots(bucket: str) -> dict[str, list[OddsSnapshot]]:
    """Every pull under `odds/`, as per-game series ordered by read time.

    Not `endgame_aws.read_all_odds`, which yields the records without the
    key they came from -- and the key is the only place the read time is.
    Same client and the same gather, so it costs what that did.
    """
    session = get_session()
    async with session.create_client("s3") as client:
        keys = [key async for key in list_keys(bucket, "odds/", client)]
        stamped = [(key, read_at_from_key(key)) for key in keys]
        bodies = await asyncio.gather(
            *(read_from_s3(bucket, key, client) for key, at in stamped if at)
        )
    history: dict[str, list[OddsSnapshot]] = {}
    for (_, read_at), body in zip((s for s in stamped if s[1]), bodies):
        assert read_at is not None
        for game in json.loads(body.decode()):
            snapshot = _parse_snapshot(game, read_at)
            if snapshot is not None:
                history.setdefault(game["competition_id"], []).append(snapshot)
    for series in history.values():
        series.sort(key=lambda s: s.read_at)
    return history


class OddsDatabase:
    """The line on every game the odds pulls have seen, plus how it got there.

    `get_odds` is the one number the rest of cassandra scores against: the
    spread at the most recent read that had one. That's a closing line for
    a played game, since ESPN drops a game's prices once it's final, and
    the current line for one that hasn't kicked off. `snapshots` is the
    whole series behind it, for `cassandra.betting`, which wants the line
    at a particular moment rather than the last one.
    """

    def __init__(
        self,
        odds: dict[str, Odds],
        history: dict[str, list[OddsSnapshot]] | None = None,
    ) -> None:
        self._odds = odds
        self._history = history or {}

    @classmethod
    async def from_s3(cls, bucket: str) -> "OddsDatabase":
        return cls.from_history(await _read_snapshots(bucket))

    @classmethod
    def from_history(cls, history: dict[str, list[OddsSnapshot]]) -> "OddsDatabase":
        """Build from per-game series already in read order."""
        odds = {}
        for game_id, series in history.items():
            # The last read with a spread on it, not the last read: a book
            # takes a line off and puts it back, and a game with no price
            # in its final pull still had one an hour earlier.
            spreads = [s.spread for s in series if s.spread is not None]
            if spreads:
                odds[game_id] = Odds(game_id=game_id, spread=spreads[-1])
        return cls(odds, history)

    def get_odds(self, game_id: str) -> Odds | None:
        return self._odds.get(game_id)

    def snapshots(self, game_id: str) -> list[OddsSnapshot]:
        """Every read of this game's prices, oldest first. Empty if never priced."""
        return self._history.get(game_id, [])

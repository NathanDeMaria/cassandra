from endgame_aws import read_all_odds
from pydantic import BaseModel


class Odds(BaseModel):
    game_id: str
    spread: float


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


class OddsDatabase:
    def __init__(self, odds: dict[str, Odds]) -> None:
        self._odds = odds

    @classmethod
    async def from_s3(cls, bucket: str) -> "OddsDatabase":
        # TODO: handle a game that's come through multiple times
        # right now, this'll just be the latest in the list,
        # which might not be the most recent
        games = (_parse_odds(g) async for g in read_all_odds(bucket, "odds/"))
        return cls({o.game_id: o async for o in games if o is not None})

    def get_odds(self, game_id: str) -> Odds | None:
        return self._odds.get(game_id)

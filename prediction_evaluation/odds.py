from endgame_aws import read_all_odds
from pydantic import BaseModel


class Odds(BaseModel):
    game_id: str
    spread: float


def _parse_odds(game: dict) -> Odds:
    return Odds(
        game_id=game["competition_id"],
        spread=game["odds"][0]["spread"],
    )


class OddsDatabase:
    def __init__(self, odds: dict[str, Odds]) -> None:
        self._odds = odds

    @classmethod
    async def from_s3(cls, bucket: str) -> "OddsDatabase":
        games = (
            _parse_odds(g) async for g in read_all_odds(bucket, "odds/")
        )
        return cls({o.game_id: o async for o in games})

    def get_odds(self, game_id: str) -> Odds | None:
        return self._odds.get(game_id)

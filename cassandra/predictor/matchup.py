"""Asking a predictor about a game that hasn't been scheduled or played."""

from datetime import datetime
from typing import NamedTuple

from .base_predictor import Predictor
from .types import Prediction


class _Matchup(NamedTuple):
    """A hypothetical game, satisfying the Matchup protocol.

    `date` and `game_id` are there because Matchup asks for them, not because
    any predictor reads them today. They're filled with something honest
    rather than left out, so a predictor that starts using them (a time decay,
    say) gets a plausible value instead of an AttributeError.
    """

    home: str
    away: str
    neutral_site: bool
    date: datetime
    game_id: str


def predict_matchup(
    predictor: Predictor,
    home: str,
    away: str,
    neutral_site: bool = False,
    date: datetime | None = None,
) -> Prediction:
    """The win probability for a matchup that isn't a Game yet.

    `predict_game` takes a Matchup -- the pre-game half of a Game -- so this
    is a small stand-in plus a call. It's here so every caller answering
    "what if these two played?" isn't inventing its own.

    Probability is the home team's, and `neutral_site` turns off the home
    advantage rather than picking a different venue.
    """
    return predictor.predict_game(
        _Matchup(
            home=home,
            away=away,
            neutral_site=neutral_site,
            date=date or datetime.now(),
            game_id=f"{away}@{home}",
        )
    )

from datetime import UTC, datetime

import pandas as pd
import pytest

from account_review import CENTRAL, in_league, replay_cutoff, slate
from cassandra.picks import FOOTBALL, MENS, WOMENS


def test_football_stands_at_the_fill() -> None:
    fill = datetime(2026, 11, 21, 17, 5, tzinfo=UTC)
    assert replay_cutoff(FOOTBALL, fill) == fill


def test_basketball_stands_on_the_mornings_data() -> None:
    # 10:15 Central: that morning's ratings were in
    morning = datetime(2026, 11, 20, 6, tzinfo=CENTRAL)
    assert replay_cutoff(WOMENS, datetime(2026, 11, 20, 16, 15, tzinfo=UTC)) == morning
    # 2:00 Central, before the night's games are in: the morning before
    assert replay_cutoff(MENS, datetime(2026, 11, 21, 8, tzinfo=UTC)) == morning


def test_a_league_is_its_series() -> None:
    frame = pd.DataFrame(
        {"event": ["KXNCAAFGAME-26NOV21YALEHARV", "KXNCAAWBGAME-26NOV21UCIUSD"]}
    )
    assert in_league(frame, WOMENS)["event"].tolist() == ["KXNCAAWBGAME-26NOV21UCIUSD"]
    assert in_league(frame, MENS).empty


def test_a_bet_with_no_price_in_a_view_takes_the_next_one_it_has() -> None:
    nan = float("nan")
    bets = pd.DataFrame(
        {
            "title": ["A @ B", "C @ D"],
            "bet_short": ["A", "D"],
            "stake": [4.0, 2.0],
            "contracts": [10.0, 3.0],
            "model": [nan, 0.7],
            "replayed": [0.45, nan],
            "market": [nan, 0.6],
            "close": [0.4, 0.65],
        }
    )
    assert [b.probability for b in slate(bets, "market")] == [0.45, 0.6]
    assert [b.probability for b in slate(bets, "replayed")] == [0.45, 0.7]
    assert [b.probability for b in slate(bets, "model")] == pytest.approx([0.45, 0.7])

import numpy as np
import pytest

from . import account_value as av


def test_a_no_pays_when_the_team_loses():
    assert av.held_probability(True, 0.7) == pytest.approx(0.7)
    assert av.held_probability(False, 0.7) == pytest.approx(0.3)


def test_the_expected_balance_is_cash_plus_probability_weighted_payouts():
    bets = [av.Bet("a", 4.0, 10.0, 0.5), av.Bet("b", 1.0, 2.0, 1.0)]
    balances = av.simulate(bets, cash=3.0, draws=50_000)
    s = av.summary(bets, 3.0, balances)
    assert s["expected"] == pytest.approx(3.0 + 5.0 + 2.0)
    assert s["profit"] == pytest.approx(10.0 - 3.0 - 5.0)
    assert s["best"] == pytest.approx(15.0)
    assert balances.mean() == pytest.approx(s["expected"], abs=0.05)
    assert balances.std() == pytest.approx(s["sd"], abs=0.05)


def test_a_decided_bet_adds_no_spread():
    bets = [av.Bet("won", 3.0, 5.0, 1.0), av.Bet("lost", 3.0, 5.0, 0.0)]
    balances = av.simulate(bets, cash=1.0, draws=1_000)
    assert set(balances) == {6.0}
    assert av.summary(bets, 1.0, balances)["sd"] == 0.0


def test_the_market_total_falls_back_to_the_model_where_there_is_no_price():
    bets = [av.Bet("a", 1.0, 10.0, 0.6, market=0.5), av.Bet("b", 1.0, 10.0, 0.9)]
    s = av.summary(bets, 0.0, av.simulate(bets, 0.0, draws=10))
    assert s["market_expected"] == pytest.approx(5.0 + 9.0)


def test_the_histogram_has_a_line_per_bin():
    lines = av.histogram(np.arange(100.0), bins=5)
    assert len(lines) == 5


def _game(game_id, hours_from_cutoff, completed=True):
    from datetime import UTC, datetime, timedelta

    from endgame.types import Game

    cutoff = datetime(2026, 10, 24, 12, tzinfo=UTC)
    return Game(
        home="H",
        home_score=21,
        away="A",
        away_score=14,
        neutral_site=False,
        completed=completed,
        date=cutoff + timedelta(hours=hours_from_cutoff),
        game_id=game_id,
        status="STATUS_FINAL",
    )


def test_a_game_not_over_by_the_cutoff_has_no_result():
    from datetime import UTC, datetime

    from endgame.types import Season, Week

    season = Season(
        weeks=[
            Week(
                games=[_game("old", -10), _game("under_way", -2), _game("next", 30)],
                number=1,
            )
        ],
        year=2026,
    )
    cut = av.as_of(season, datetime(2026, 10, 24, 12, tzinfo=UTC))
    old, under_way, later = cut.weeks[0].games
    assert old.completed and old.home_score == 21
    assert not under_way.completed and under_way.home_score == 0
    assert not later.completed


def test_the_actual_result_is_placed_within_the_draws():
    bets = [av.Bet("a", 4.0, 10.0, 0.5), av.Bet("b", 1.0, 2.0, 1.0)]
    # a pays: 12 in; a loses: 2 in. Staked 5.
    won = av.expectation(bets, 12.0, draws=50_000)
    lost = av.expectation(bets, 2.0, draws=50_000)
    assert won["actual"] == pytest.approx(7.0)
    assert lost["actual"] == pytest.approx(-3.0)
    assert won["percentile"] == pytest.approx(0.75, abs=0.02)
    assert lost["percentile"] == pytest.approx(0.25, abs=0.02)
    assert won["expected"] == pytest.approx(2.0)

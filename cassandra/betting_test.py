from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from .betting import (
    GAME_LENGTH,
    LineColumns,
    Record,
    american_payout,
    american_to_probability,
    clv_by_edge,
    line_windows,
    moneyline_bets,
    moneyline_calibration,
    no_vig_home_probability,
    previous_game_end,
    record,
    spread_bets,
)
from .odds import OddsDatabase, OddsSnapshot

KICKOFF = datetime(2026, 9, 12, 16, 0, tzinfo=UTC)


def _game(game_id, home, away, date, home_score=0, away_score=0, prob=0.5, margin=0.0):
    return dict(
        game_id=game_id,
        home_team=home,
        away_team=away,
        date=date,
        home_score=home_score,
        away_score=away_score,
        team1_win_prob=prob,
        predicted_margin=margin,
        year=date.year,
    )


def _snapshot(hours_before, spread, home_ml=None, away_ml=None, kickoff=KICKOFF):
    return OddsSnapshot(
        read_at=kickoff - timedelta(hours=hours_before),
        spread=spread,
        home_moneyline=home_ml,
        away_moneyline=away_ml,
    )


def test_american_prices_convert_both_ways() -> None:
    prices = pd.Series([-150.0, 200.0, -100.0, np.nan])
    assert american_to_probability(prices).tolist()[:3] == pytest.approx(
        [0.6, 1 / 3, 0.5]
    )
    assert american_payout(prices).tolist()[:3] == pytest.approx([2 / 3, 2.0, 1.0])
    assert np.isnan(american_to_probability(prices).iloc[3])


def test_no_vig_splits_the_hold_proportionally() -> None:
    # -110 both sides is a coin flip once the hold is out.
    even = no_vig_home_probability(pd.Series([-110.0]), pd.Series([-110.0]))
    assert even.iloc[0] == pytest.approx(0.5)
    # -200 / +170: 0.667 and 0.370 imply a 3.7% hold; the home share is 0.643.
    lopsided = no_vig_home_probability(pd.Series([-200.0]), pd.Series([170.0]))
    assert lopsided.iloc[0] == pytest.approx((2 / 3) / (2 / 3 + 1 / 2.7))


def test_previous_game_end_is_the_later_previous_game_plus_a_game() -> None:
    week1 = datetime(2026, 9, 5, 16, 0, tzinfo=UTC)
    week1_late = datetime(2026, 9, 6, 0, 0, tzinfo=UTC)
    games = pd.DataFrame(
        [
            _game("a", "X", "Y", week1),
            _game("b", "Z", "W", week1_late),
            _game("c", "X", "Z", KICKOFF),
        ]
    )
    ends = previous_game_end(games)
    # X last played at 16:00, Z at 00:00 the next day: the window opens after Z's.
    assert ends["c"] == week1_late + GAME_LENGTH
    # A team's first game has no previous one, however many the opponent has.
    assert pd.isna(ends["a"])
    assert pd.isna(ends["b"])


def _one_lined_game(snapshots, **game):
    games = pd.DataFrame(
        [
            _game("prev", "X", "Q", KICKOFF - timedelta(days=7)),
            _game("prev2", "Y", "R", KICKOFF - timedelta(days=7)),
            _game("g", "X", "Y", KICKOFF, **game),
        ]
    )
    return line_windows(games, OddsDatabase({}, {"g": snapshots}))


def test_line_windows_picks_first_read_after_previous_game_and_last_before_kickoff() -> (
    None
):
    lines = _one_lined_game(
        [
            # Read during the previous week's games: the model hadn't seen them.
            _snapshot(hours_before=24 * 7 - 1, spread=-3.0),
            _snapshot(hours_before=24 * 6, spread=-4.0, home_ml=-180.0, away_ml=150.0),
            _snapshot(hours_before=24, spread=-6.0),
            _snapshot(hours_before=1, spread=-7.0, home_ml=-260.0, away_ml=210.0),
            # Read after kickoff: not a line anyone could bet.
            _snapshot(hours_before=-2, spread=-10.0),
        ]
    )
    assert len(lines) == 1
    row = lines.iloc[0]
    assert row["game_id"] == "g"
    assert row[LineColumns.ENTRY_SPREAD] == -4.0
    assert row[LineColumns.ENTRY_HOME_ML] == -180.0
    assert row[LineColumns.CLOSE_SPREAD] == -7.0
    assert row[LineColumns.CLOSE_AWAY_ML] == 210.0
    assert row[LineColumns.CLOSE_READ_AT] == KICKOFF - timedelta(hours=1)


def test_a_game_only_read_before_its_previous_game_has_a_close_but_no_entry() -> None:
    lines = _one_lined_game([_snapshot(hours_before=24 * 7 - 1, spread=-3.0)])
    assert lines[LineColumns.CLOSE_SPREAD].tolist() == [-3.0]
    assert lines[LineColumns.ENTRY_SPREAD].isna().all()
    # And with nothing before kickoff at all, the game isn't in the table.
    assert _one_lined_game([_snapshot(hours_before=-1, spread=-3.0)]).empty


def test_first_game_of_the_frame_has_a_close_and_no_entry() -> None:
    games = pd.DataFrame([_game("g", "X", "Y", KICKOFF)])
    lines = line_windows(games, OddsDatabase({}, {"g": [_snapshot(1, -3.0)]}))
    assert lines[LineColumns.CLOSE_SPREAD].tolist() == [-3.0]
    assert lines[LineColumns.ENTRY_SPREAD].isna().all()


def _lines(*rows):
    """A graded frame straight from (entry, close, margin, mov) tuples."""
    return pd.DataFrame(
        [
            {
                "game_id": str(i),
                "home_score": mov if mov > 0 else 0,
                "away_score": -mov if mov < 0 else 0,
                "predicted_margin": margin,
                "team1_win_prob": 0.5,
                LineColumns.ENTRY_SPREAD: entry,
                LineColumns.CLOSE_SPREAD: close,
            }
            for i, (entry, close, margin, mov) in enumerate(rows)
        ]
    )


def test_clv_is_the_move_toward_the_bet_side() -> None:
    bets = spread_bets(
        _lines(
            # Model likes home at -3; line closes -6: home got 3 points dearer.
            (-3.0, -6.0, +5.0, 10),
            # Model likes away at -3; line closes -6: away got 3 points cheaper.
            (-3.0, -6.0, -1.0, 10),
            # Model likes away at -3; line closes -1: away's number shrank.
            (-3.0, -1.0, -1.0, 10),
        )
    )
    assert bets["bet_home"].tolist() == [True, False, False]
    assert bets["clv"].tolist() == [3.0, -3.0, 2.0]
    assert bets["edge_points"].tolist() == [2.0, 4.0, 4.0]


def test_pushes_are_neither_win_nor_loss() -> None:
    bets = spread_bets(
        _lines(
            (-3.0, -3.0, +5.0, 3),  # home by exactly the number
            (-3.0, -2.5, +5.0, 3),  # ... but covers the close
            (-3.0, -3.0, +5.0, 7),
            (-3.0, -3.0, +5.0, -1),
        )
    )
    assert record(bets, "entry") == Record(wins=1, losses=1, pushes=2)
    assert record(bets, "close") == Record(wins=2, losses=1, pushes=1)
    assert str(record(bets, "entry")) == "1-1-2"
    assert record(bets, "entry").cover_rate == 0.5


def test_games_missing_either_spread_are_not_graded() -> None:
    lines = _lines((-3.0, -3.0, +5.0, 7), (None, -3.0, +5.0, 7), (-3.0, None, 1.0, 7))
    assert spread_bets(lines)["game_id"].tolist() == ["0"]


def test_clv_by_edge_buckets_by_disagreement() -> None:
    bets = spread_bets(
        _lines(
            (-3.0, -4.0, +4.0, 10),  # edge 1, clv +1, covered
            (-3.0, -2.0, +8.0, 1),  # edge 5, clv -1, lost
            (-3.0, -3.0, +20.0, 3),  # edge 17, clv 0, push
        )
    )
    table = clv_by_edge(bets).set_index("edge_points")
    assert table.index.tolist() == ["0-2", "4-7", "7+"]
    assert table.loc["0-2", "clv_mean"] == 1.0
    assert table.loc["0-2", "record_entry"] == "1-0-0"
    assert table.loc["4-7", "clv_negative"] == 1.0
    assert table.loc["7+", "record_entry"] == "0-0-1"
    assert pd.isna(table.loc["7+", "cover_rate_entry"])


def _priced(*rows):
    """(model prob, home ml, away ml, home won) rows, priced at the close."""
    return pd.DataFrame(
        [
            {
                "game_id": str(i),
                "team1_win_prob": prob,
                "home_score": 1 if home_won else 0,
                "away_score": 0 if home_won else 1,
                LineColumns.CLOSE_HOME_ML: home,
                LineColumns.CLOSE_AWAY_ML: away,
            }
            for i, (prob, home, away, home_won) in enumerate(rows)
        ]
    )


def test_calibration_compares_model_and_market_on_the_same_games() -> None:
    lines = _priced(
        (0.9, -300.0, 250.0, True),
        (0.5, -110.0, -110.0, False),
        (0.5, None, 150.0, True),  # unpriced on one side: left out of both
    )
    cal = moneyline_calibration(lines, "close")
    assert cal["n"] == 2
    assert cal["brier_model"] == pytest.approx(((0.9 - 1) ** 2 + 0.5**2) / 2)
    market_first = (3 / 4) / (3 / 4 + 1 / 3.5)
    assert cal["brier_market"] == pytest.approx(((market_first - 1) ** 2 + 0.5**2) / 2)
    assert cal["hold"] == pytest.approx(
        ((3 / 4 + 1 / 3.5 - 1) + (2 * 110 / 210 - 1)) / 2
    )


def test_flat_strategy_bets_the_model_side_past_the_threshold() -> None:
    lines = _priced(
        # Pick'em at even money; model 60% home; home wins: +1.
        (0.6, 100.0, 100.0, True),
        # Pick'em; model 40% home -> away; home wins: -1.
        (0.4, 100.0, 100.0, True),
        # No-vig market 72.4% home; model 74%: a 0.016 edge on the favorite;
        # home wins at -300: +1/3.
        (0.74, -300.0, 250.0, True),
    )
    table = moneyline_bets(lines, "close").set_index("strategy")
    any_edge = table.loc["flat, edge > 0.00"]
    assert any_edge["n"] == 3
    assert any_edge["units"] == pytest.approx(1 / 3)
    # A favorite is a negative price; neither side of a pick'em is one.
    assert any_edge["n_favorites"] == 1
    assert any_edge["roi_favorites"] == pytest.approx(1 / 3)
    assert any_edge["roi_underdogs"] == pytest.approx(0.0)
    # Only the 10-point edges clear 0.05; the favorite bet drops out.
    five = table.loc["flat, edge > 0.05"]
    assert five["n"] == 2
    assert five["hit_rate"] == 0.5
    assert table.loc["flat, edge > 0.15", "n"] == 0
    assert pd.isna(table.loc["flat, edge > 0.15", "roi"])


def test_kelly_stakes_the_fraction_of_the_edge_and_skips_no_edge() -> None:
    lines = _priced(
        # p=0.6 at even money: f = (0.6*1 - 0.4)/1 = 0.2, a quarter is 0.05.
        (0.6, 100.0, -120.0, True),
        # No edge on either side against a 50/50 market: no stake.
        (0.5, -110.0, -110.0, True),
    )
    kelly = (
        moneyline_bets(lines, "close")
        .set_index("strategy")
        .loc["kelly x0.25, any edge"]
    )
    assert kelly["n"] == 1
    assert kelly["units"] == pytest.approx(0.05)
    assert kelly["roi"] == pytest.approx(1.0)


def test_market_baselines_take_every_favorite_or_every_underdog() -> None:
    lines = _priced((0.5, -300.0, 250.0, True), (0.5, -300.0, 250.0, False))
    table = moneyline_bets(lines, "close").set_index("strategy")
    assert table.loc["every favorite", "units"] == pytest.approx(1 / 3 - 1)
    assert table.loc["every underdog", "units"] == pytest.approx(2.5 - 1)
    assert table.loc["every favorite", "n_favorites"] == 2

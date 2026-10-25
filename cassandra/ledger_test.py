import math
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from . import ledger
from .placing import CENTRAL

EVENT = "KXNCAAFGAME-26OCT16OKSTHOU"
EVENTS = {
    EVENT: {
        "event": {"title": "Oklahoma St. vs Houston"},
        "markets": [
            {"ticker": f"{EVENT}-OKST", "yes_sub_title": "Oklahoma St."},
            {"ticker": f"{EVENT}-HOU", "yes_sub_title": "Houston"},
        ],
    }
}
CODES = ["OKST", "HOU"]


def _fill(ticker: str, order_id: str, at: str) -> dict:
    return {"ticker": ticker, "order_id": order_id, "created_time": at}


def test_a_game_bet_through_both_markets_is_one_bet_on_one_team():
    # YES on Houston and NO on Oklahoma State both pay if Houston wins.
    settlements = [
        {
            "ticker": f"{EVENT}-HOU",
            "yes_count_fp": "7.00",
            "yes_total_cost_dollars": "4.13",
            "no_count_fp": "0.00",
            "no_total_cost_dollars": "0",
            "revenue": 700,
            "fee_cost": "0.12",
        },
        {
            "ticker": f"{EVENT}-OKST",
            "yes_count_fp": "0.00",
            "yes_total_cost_dollars": "0",
            "no_count_fp": "7.00",
            "no_total_cost_dollars": "4.06",
            "revenue": 700,
            "fee_cost": "0.11",
        },
    ]
    fills = [
        _fill(f"{EVENT}-OKST", "o2", "2026-10-11T04:27:00Z"),
        _fill(f"{EVENT}-HOU", "o1", "2026-10-11T04:15:00Z"),
    ]
    bets = ledger.bets(fills, settlements, [], EVENTS)
    assert len(bets) == 1
    bet = bets.iloc[0]
    assert bet["bet_code"] == "HOU" and bet["bet_short"] == "Houston"
    assert bet["contracts"] == 14
    assert bet["avg_price"] == pytest.approx(8.19 / 14)
    assert bet["stake"] == pytest.approx(8.19 + 0.23)
    assert bet["settled"] and bet["won"]
    assert bet["pnl"] == pytest.approx(14 - 8.19 - 0.23)
    assert bet["first_fill"] == pd.Timestamp("2026-10-11T04:15:00Z")
    assert bet["order_ids"] == ["o1", "o2"]


def test_a_position_sold_before_the_end_is_paid_what_it_sold_for():
    # 11 YES on Houston at 0.88, sold at 0.97: Kalshi books the sale as 11
    # NO at 0.03, and the settlement's revenue is 0.
    settlements = [
        {
            "ticker": f"{EVENT}-HOU",
            "yes_count_fp": "11.00",
            "yes_total_cost_dollars": "9.68",
            "no_count_fp": "11.00",
            "no_total_cost_dollars": "0.33",
            "revenue": 0,
            "fee_cost": "0.10",
        }
    ]
    bet = ledger.bets([], settlements, [], EVENTS).iloc[0]
    assert bet["bet_code"] == "HOU"
    assert bet["settled"] and bet["won"]
    assert bet["pnl"] == pytest.approx(11 - 9.68 - 0.33 - 0.10)


def test_an_open_no_position_is_a_bet_on_the_other_team():
    positions = [
        {
            "ticker": f"{EVENT}-OKST",
            "position_fp": "-15.00",
            "market_exposure_dollars": "9.29",
            "fees_paid_dollars": "0.25",
        }
    ]
    bet = ledger.bets([], [], positions, EVENTS).iloc[0]
    assert bet["bet_code"] == "HOU"
    assert not bet["settled"] and bet["won"] is None
    assert bet["avg_price"] == pytest.approx(9.29 / 15)
    assert math.isnan(bet["pnl"])


def _bet(**kw) -> dict:
    return {
        "event": EVENT,
        "bet_code": "HOU",
        "order_ids": [],
        "first_fill": pd.Timestamp("2026-10-11T04:15:00Z"),
        **kw,
    }


def _run(model: float, game_id: str = "401") -> pd.DataFrame:
    # A run that picked Oklahoma State, by buying NO on Houston.
    return pd.DataFrame(
        [
            {
                "event": EVENT,
                "buy": f"NO {EVENT}-HOU",
                "bet": "Oklahoma State Cowboys",
                "bet_home": False,
                "game": "Oklahoma State Cowboys @ Houston Cougars",
                "game_id": game_id,
                "kickoff": "2026-10-17 18:30:00-05:00",
                "divisions": "FBS v FBS",
                "model": model,
                "market": 0.42,
            }
        ]
    )


def test_read_runs_stacks_the_picks_runs_by_when_they_started(tmp_path):
    _run(0.30).assign(skip="not asked for").to_csv(
        tmp_path / "ncaafb_m_20261011-1609.csv", index=False
    )
    _run(0.25).to_csv(tmp_path / "ncaafb_m_20261010-2309.csv", index=False)
    pd.DataFrame({"order_id": ["o1"]}).to_csv(
        tmp_path / "ncaafb_m_20261010-2309_orders.csv", index=False
    )
    runs = ledger.read_runs(tmp_path)
    assert list(runs["model"]) == [0.25, 0.30]
    assert runs["started"].iloc[0] == pd.Timestamp("2026-10-10 23:09", tz=CENTRAL)
    assert "skip" not in runs and runs["game_id"].iloc[0] == "401"
    assert ledger.read_runs(tmp_path / "none").empty


def test_bet_time_turns_a_run_to_the_bets_side_and_takes_the_run_before_the_fill():
    runs = pd.concat(
        [
            _run(0.25).assign(started=pd.Timestamp("2026-10-10 23:09", tz=CENTRAL)),
            # Started after the fill: it names the game, but its model came too late.
            _run(0.30, game_id="402").assign(
                started=pd.Timestamp("2026-10-11 16:09", tz=CENTRAL)
            ),
        ],
        ignore_index=True,
    )
    found = ledger.bet_time(_bet(), CODES, runs, pd.DataFrame())
    assert found["model"] == pytest.approx(0.75)
    assert found["market"] == pytest.approx(0.58)
    assert found["model_from"] == "picks run 10-10 23:09"
    assert found["bet_home"] is True
    assert found["bet"] == "Houston Cougars"
    assert found["game_id"] == "402"


def test_bet_time_prefers_the_orders_log_weighted_by_contracts():
    orders = pd.DataFrame(
        [
            {
                "order_id": "o1",
                "ticker": f"{EVENT}-HOU",
                "action": "yes",
                "filled": 3.0,
                "model": 0.70,
                "edge": 0.10,
                "bet": "Houston Cougars",
                "game": "Oklahoma State Cowboys @ Houston Cougars",
                "game_id": "401",
            },
            {
                "order_id": "o2",
                "ticker": f"{EVENT}-OKST",
                "action": "no",
                "filled": 1.0,
                "model": 0.80,
                "edge": 0.20,
                "bet": "Houston Cougars",
                "game": "Oklahoma State Cowboys @ Houston Cougars",
                "game_id": "401",
            },
            # Never filled: says nothing about what was bought.
            {
                "order_id": "o3",
                "ticker": f"{EVENT}-HOU",
                "action": "yes",
                "filled": 0.0,
                "model": 0.99,
                "edge": 0.5,
                "bet": "Houston Cougars",
                "game": "Oklahoma State Cowboys @ Houston Cougars",
                "game_id": "401",
            },
        ]
    )
    no_runs = ledger.read_runs(Path("/nowhere"))
    found = ledger.bet_time(_bet(order_ids=["o1", "o2", "o3"]), CODES, no_runs, orders)
    assert found["model_from"] == "orders log"
    assert found["model"] == pytest.approx((3 * 0.70 + 0.80) / 4)
    assert found["market"] == pytest.approx((3 * 0.60 + 0.60) / 4)
    assert found["bet_home"] is True


def _candle(end: int, bid: float, ask: float) -> dict:
    return {
        "end_period_ts": end,
        "yes_bid": {"close_dollars": f"{bid:.4f}"},
        "yes_ask": {"close_dollars": f"{ask:.4f}"},
    }


def test_no_vig_is_the_last_candle_by_then_scaled_to_one():
    candles = {
        "HOU": [
            _candle(100, 0.50, 0.54),
            _candle(200, 0.60, 0.64),
            _candle(300, 0.9, 0.95),
        ],
        "OKST": [
            _candle(100, 0.44, 0.48),
            _candle(200, 0.38, 0.42),
            _candle(300, 0.05, 0.1),
        ],
    }
    found = ledger.no_vig(candles, 250)
    assert found["HOU"] == pytest.approx(0.62 / 1.02)
    assert sum(found.values()) == pytest.approx(1)
    assert ledger.no_vig({"HOU": [], "OKST": candles["OKST"]}, 250) == {}


def _stats(yards: int, rush: int, rush_att: int, comp_att: str, turnovers: int) -> list:
    return [
        {"name": "totalYards", "displayValue": str(yards)},
        {"name": "rushingYards", "displayValue": str(rush)},
        {"name": "rushingAttempts", "displayValue": str(rush_att)},
        {"name": "completionAttempts", "displayValue": comp_att},
        {"name": "turnovers", "displayValue": str(turnovers)},
        {"name": "thirdDownEff", "displayValue": "5-12"},
    ]


def _passing(team_id: str, *lines: tuple[str, str]) -> dict:
    return {
        "team": {"id": team_id},
        "statistics": [
            {
                "name": "passing",
                "athletes": [
                    {
                        "athlete": {"displayName": name},
                        "stats": [comp_att, "200", "7.0", "1", "0", "50"],
                    }
                    for name, comp_att in lines
                ],
            }
        ],
    }


def _play(period: int, clock: int, away: int, home: int, text: str = "TD", team="HOU"):
    return {
        "period": {"number": period},
        "clock": {"value": clock, "displayValue": f"{clock // 60}:{clock % 60:02d}"},
        "awayScore": away,
        "homeScore": home,
        "team": {"abbreviation": team},
        "text": text,
    }


def _summary(scoring, home_score, away_score, wp, home_stats, away_stats, players=()):
    return {
        "header": {
            "competitions": [
                {
                    "status": {"period": max(p["period"]["number"] for p in scoring)},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": str(home_score),
                            "team": {"id": "1", "abbreviation": "HOU"},
                        },
                        {
                            "homeAway": "away",
                            "score": str(away_score),
                            "team": {"id": "2", "abbreviation": "OKST"},
                        },
                    ],
                }
            ]
        },
        "scoringPlays": scoring,
        "winprobability": [{"homeWinPercentage": w} for w in wp],
        "boxscore": {
            "teams": [
                {"homeAway": "home", "statistics": home_stats},
                {"homeAway": "away", "statistics": away_stats},
            ],
            "players": list(players),
        },
        "pickcenter": [{"details": "HOU -4"}],
        "article": {
            "headline": "Houston rallies",
            "story": "<p>Houston came back from two scores down to win it late.</p>",
        },
    }


COMEBACK = _summary(
    [
        _play(1, 600, 7, 0, team="OKST"),
        _play(2, 300, 14, 0, team="OKST"),
        _play(3, 400, 14, 7),
        _play(4, 600, 14, 14),
        _play(4, 120, 14, 21),
    ],
    21,
    14,
    [0.6, 0.4, 0.1, 0.3, 0.5, 0.9, 0.2],
    _stats(400, 200, 40, "20/30", 1),
    _stats(300, 80, 30, "15/30", 2),
)


def test_a_late_comeback_from_the_winners_side():
    f = ledger.facts(COMEBACK, bet_home=True)
    assert f["score"] == "HOU 21–14 OKST"
    assert f["margin"] == 7 and not f["overtime"]
    assert f["largest_deficit"] == -14 and f["half_margin"] == -14
    assert f["go_ahead"] == "Q4 2:00" and f["late_decider"]
    assert f["wp_low"] == pytest.approx(0.1)  # the last entry is left out
    assert f["ypp_for"] == pytest.approx(400 / 70)
    tags = ledger.tags(f, won=True)
    assert {"one_score", "late_go_ahead", "comeback"} <= set(tags)
    assert ledger.box_read(f, won=True) == "earned"
    assert ledger.suggest(f, tags, won=True, price=0.6) == "late_swing"
    assert "came back" in f["recap"]


def test_the_same_game_from_the_losers_side_has_the_same_game_tags():
    f = ledger.facts(COMEBACK, bet_home=False)
    assert f["margin"] == -7 and f["largest_lead"] == 14
    tags = ledger.tags(f, won=False)
    assert {"one_score", "late_go_ahead", "comeback"} <= set(tags)
    assert ledger.box_read(f, won=False) == "deserved"


def test_a_blowouts_backup_quarterback_doesnt_make_it_a_qb_game():
    rout = _summary(
        [_play(1, 600, 0, 7), _play(2, 300, 0, 28), _play(4, 300, 7, 42, team="OKST")],
        42,
        7,
        [0.8, 0.9, 0.99, 1.0],
        _stats(550, 250, 40, "20/25", 0),
        _stats(250, 60, 25, "20/40", 2),
        players=[
            _passing("2", ("Starter", "12/25"), ("Backup", "8/15")),
            _passing("1", ("Ace", "20/25")),
        ],
    )
    f = ledger.facts(rout, bet_home=True)
    tags = ledger.tags(f, won=True)
    assert {"blowout", "wire_to_wire", "run_game", "qb_change_them"} <= set(tags)
    assert "qb_change_us" not in tags
    assert ledger.suggest(f, tags, won=True, price=0.8) == "outclassed"


def test_return_touchdowns_are_counted_for_the_side_that_scored_them():
    game = _summary(
        [
            _play(
                1, 600, 7, 0, "Joe Smith 80 Yd Interception Return (Kick)", team="OKST"
            ),
            _play(2, 300, 7, 3, "FG"),
        ],
        3,
        7,
        [0.5, 0.4, 0.3],
        _stats(300, 100, 30, "20/30", 2),
        _stats(250, 100, 30, "15/30", 0),
    )
    f = ledger.facts(game, bet_home=True)
    assert f["return_tds_against"] == 1 and f["return_tds_for"] == 0
    tags = ledger.tags(f, won=False)
    assert "return_td" in tags
    assert ledger.suggest(f, tags, won=False, price=0.6) == "turnovers"


@pytest.mark.parametrize(
    "yards_for, ypp_for, won, read",
    [
        (450, 7.0, True, "earned"),
        (250, 4.0, True, "lucky"),
        (450, 7.0, False, "unlucky"),
        (250, 4.0, False, "deserved"),
        # More yards on more plays but worse per play: the box score is split.
        (450, 4.5, True, "even"),
        (360, 7.0, True, "even"),
    ],
)
def test_box_read_needs_yards_and_yards_a_play_to_agree(yards_for, ypp_for, won, read):
    f = {
        "yards_for": yards_for,
        "yards_against": 350,
        "ypp_for": ypp_for,
        "ypp_against": 5.5,
    }
    assert ledger.box_read(f, won) == read


@pytest.mark.parametrize(
    "kickoff, saturday",
    [
        ("2026-10-15 19:00", date(2026, 10, 17)),  # Thursday
        ("2026-10-17 21:30", date(2026, 10, 17)),  # Saturday night
        ("2026-10-18 12:00", date(2026, 10, 17)),  # Sunday
        ("2026-09-07 19:00", date(2026, 9, 5)),  # Labor Day Monday
        ("2026-10-20 18:00", date(2026, 10, 24)),  # Tuesday
    ],
)
def test_a_week_runs_tuesday_to_monday(kickoff, saturday):
    when = pd.Timestamp(kickoff, tz=CENTRAL).to_pydatetime()
    assert ledger.week_of(when) == saturday
    assert ledger.week_of(when.astimezone(UTC)) == saturday


def test_themes_must_use_a_known_type(tmp_path):
    path = tmp_path / "themes.toml"
    path.write_text(
        f'[{EVENT}]\ntype = "late_swing"\ntheme = "Walk-off"\nwhy = "A kick at the gun."\n'
    )
    assert ledger.read_themes(path)[EVENT] == ledger.Theme(
        "late_swing", "Walk-off", "A kick at the gun."
    )
    path.write_text(f'[{EVENT}]\ntype = "bad luck"\n')
    with pytest.raises(ValueError, match="bad luck"):
        ledger.read_themes(path)
    assert ledger.read_themes(tmp_path / "missing.toml") == {}


def _row(event: str, won: bool | None, price: float, contracts: float, **kw) -> dict:
    settled = won is not None
    cost = price * contracts
    payout = (contracts if won else 0.0) if settled else math.nan
    return {
        "event": event,
        "title": f"{event} game",
        "bet_short": event,
        "settled": settled,
        "won": won,
        "contracts": contracts,
        "avg_price": price,
        "cost": cost,
        "fees": 0.1,
        "stake": cost + 0.1,
        "payout": payout,
        "pnl": payout - cost - 0.1 if settled else math.nan,
        "week": date(2026, 10, 17) if settled else date(2026, 10, 24),
        "kickoff": pd.Timestamp("2026-10-17 18:00", tz=CENTRAL),
        "divisions": "FBS v FBS",
        "bet_home": True,
        "model": price + 0.1,
        "market": price,
        "close": price + 0.05 if settled else math.nan,
        "mark": math.nan if settled else price + 0.02,
        "score": "A 1–0 B" if settled else None,
        "read": "earned" if won else "unlucky",
        "tags": ["one_score"] if settled else [],
        "type": "late_swing" if settled else None,
        "type_from": "theme" if won else "box score",
        "theme": "Walk-off",
        "why": "A kick at the gun.",
        **kw,
    }


LEDGER = pd.DataFrame(
    [
        _row("WIN", True, 0.40, 10),
        _row("LOSS", False, 0.60, 10),
        _row("OPEN", None, 0.50, 4),
    ]
)


def test_scorecard():
    settled = LEDGER[LEDGER["settled"]].assign(won=lambda f: f["won"].astype(bool))
    card = ledger.scorecard(settled)
    assert (card["wins"], card["losses"]) == (1, 1)
    assert card["staked"] == pytest.approx(10.2)
    assert card["pnl"] == pytest.approx(10 - 10.2)
    assert card["model_ev"] == pytest.approx(10 * 0.1 * 2 - 0.2)
    assert card["close_ev"] == pytest.approx(10 * 0.05 * 2 - 0.2)
    assert card["clv"] == pytest.approx(5)
    assert card["beat_close"] == 2
    assert card["wins_model"] == pytest.approx(1.2)


def test_report_has_each_part():
    page = ledger.report(LEDGER, datetime(2026, 10, 21, 12, tzinfo=UTC))
    assert "3 games bet, 2 settled over 1 week, 1 open" in page
    assert "| late_swing | 2 | 1–1 |" in page
    assert "## Week of Oct 17: 1–1" in page
    assert "**Walk-off.** A kick at the gun." in page
    # A game no one has written up shows its guessed type and its tags.
    assert "late_swing? | unlucky | _pending_ (one_score)" in page
    assert "## Open: 1 games, $2.10 staked" in page


def test_wins_losses_lists_each_settled_bet_under_its_result():
    page = ledger.wins_losses(LEDGER, datetime(2026, 10, 21, 12, tzinfo=UTC))
    wins, losses = page.split("## Losses")
    assert "## Wins (1)" in wins and "**Walk-off.** A kick at the gun." in wins
    assert "_pending_ (one_score)" in losses
    # The open bet is in neither.
    assert page.count("| Oct 17 |") == 2

"""Tests for the team-season estimates.

Most of these run on a simulated league with the effects planted: every
team-season gets a true error drawn from N(0, tau), every game's residual is
the home side's minus the away side's plus N(0, sigma) noise, and the
estimators are asked to find what was planted. A test that planted nothing
checks the null the same way.
"""

import numpy as np
import pandas as pd
import pytest

from .residuals import MARGIN_RESIDUAL, MARKET_MARGIN, PREDICTED_MARGIN
from .team_seasons import (
    game_log,
    group_season_table,
    label_tiers,
    null_counts,
    season_effects,
    select,
    summarize,
    team_games,
    team_season_table,
)

SCALE = 15.0


def league(
    teams: int = 40,
    seasons: int = 6,
    rounds: int = 12,
    tau: float = 3.0,
    sigma: float = 14.0,
    seed: int = 0,
    effects: dict[tuple[str, int], float] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], float]]:
    """A scored frame, and the team-season effects planted in it.

    Each season is `rounds` weeks of random pairings, so every team plays
    `rounds` games against a schedule that connects the whole league.
    `effects` overrides the drawn ones for the team-seasons it names.
    """
    rng = np.random.default_rng(seed)
    names = [f"team{i:02d}" for i in range(teams)]
    planted = {
        (team, 2010 + s): float(rng.normal(0, tau))
        for s in range(seasons)
        for team in names
    }
    planted |= effects or {}
    rows = []
    for s in range(seasons):
        year = 2010 + s
        for week in range(1, rounds + 1):
            order = rng.permutation(teams)
            for home, away in zip(order[::2], order[1::2]):
                h, a = names[home], names[away]
                predicted = float(rng.normal(0, 10))
                residual = (
                    planted[(h, year)] - planted[(a, year)] + rng.normal(0, sigma)
                )
                mov = round(predicted + residual)
                rows.append(
                    {
                        "game_id": f"{year}-{week}-{h}",
                        "year": year,
                        "week_number": week,
                        "home_team": h,
                        "away_team": a,
                        "home_score": 30 + max(mov, 0),
                        "away_score": 30 + max(-mov, 0),
                        "neutral_site": False,
                        "team1_win_prob": 1 / (1 + np.exp(-predicted / SCALE)),
                        PREDICTED_MARGIN: predicted,
                        MARKET_MARGIN: np.nan,
                        "spread": np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(
        frame["year"].astype(str) + "-09-01", utc=True
    ) + pd.to_timedelta(frame["week_number"] * 7, unit="D")
    frame[MARGIN_RESIDUAL] = (
        frame["home_score"] - frame["away_score"] - frame[PREDICTED_MARGIN]
    )
    return frame, planted


def test_team_games_signs_everything_to_the_team() -> None:
    frame, _ = league(teams=4, seasons=1, rounds=1)
    games = team_games(frame)
    first = frame.iloc[0]
    home = games[(games["game_id"] == first["game_id"]) & games["home_side"]].iloc[0]
    away = games[(games["game_id"] == first["game_id"]) & ~games["home_side"]].iloc[0]
    assert home["team"] == first["home_team"]
    assert home["residual"] == pytest.approx(first[MARGIN_RESIDUAL])
    assert away["residual"] == pytest.approx(-first[MARGIN_RESIDUAL])
    assert away["predicted"] == pytest.approx(-first[PREDICTED_MARGIN])
    assert home["site"] == "home" and away["site"] == "away"


def test_team_games_counts_each_team_s_games_in_order() -> None:
    frame, _ = league(teams=6, seasons=2, rounds=5)
    games = team_games(frame)
    for _, rows in games.groupby(["team", "year"]):
        assert rows["game_number"].tolist() == list(range(1, len(rows) + 1))
        assert rows["date"].is_monotonic_increasing


def test_the_ridge_finds_the_planted_spread_and_the_effects() -> None:
    frame, planted = league(teams=60, seasons=8, rounds=14, tau=3.0, sigma=12.0)
    fit = season_effects(
        frame, frame["home_team"].to_numpy(), frame["away_team"].to_numpy()
    )
    assert fit.tau == pytest.approx(3.0, abs=0.6)
    assert fit.sigma == pytest.approx(12.0, rel=0.05)
    effects = fit.effects.set_index(["label", "year"])["shrunk"]
    truth = pd.Series(planted).reindex(effects.index)
    # Shrunk toward zero, so a correlation rather than equality; with
    # reliability around 0.45 at these sizes it should be well above 0.5.
    assert np.corrcoef(effects, truth)[0, 1] > 0.55


def test_with_nothing_planted_tau_is_near_zero() -> None:
    frame, _ = league(teams=60, seasons=8, rounds=14, tau=0.0, sigma=12.0)
    fit = season_effects(
        frame, frame["home_team"].to_numpy(), frame["away_team"].to_numpy()
    )
    assert fit.tau < 1.0
    assert fit.effects["shrunk"].abs().max() < 1.0


def test_the_ridge_nets_out_an_opponent_s_error() -> None:
    # One team is 12 points worse than the model thinks. Every team that
    # played it has its raw mean inflated by 12 per game against it, and
    # none of that is theirs.
    effects = {(f"team{i:02d}", 2010): 0.0 for i in range(20)}
    effects[("team00", 2010)] = -12.0
    frame, _ = league(
        teams=20, seasons=1, rounds=19, tau=0.0, sigma=0.5, effects=effects
    )
    table = team_season_table(frame).table.set_index("team")
    against = pd.concat(
        [
            frame.loc[frame["away_team"] == "team00", "home_team"],
            frame.loc[frame["home_team"] == "team00", "away_team"],
        ]
    ).value_counts()
    opponents = table.drop(index="team00").assign(
        meetings=lambda t: against.reindex(t.index).fillna(0)
    )
    # Effects are relative to the season's average team, so team00's -12
    # comes out as -12 * 19/20 and everyone else carries its +0.6 share.
    assert table.loc["team00", "shrunk"] == pytest.approx(-11.4, abs=1.0)
    assert np.corrcoef(opponents["raw"], opponents["meetings"])[0, 1] > 0.8
    assert abs(np.corrcoef(opponents["shrunk"], opponents["meetings"])[0, 1]) < 0.5


def test_the_table_is_sorted_and_carries_early_and_late() -> None:
    frame, _ = league()
    result = team_season_table(frame, early_games=4)
    assert result.table["shrunk"].is_monotonic_decreasing
    row = result.table.iloc[0]
    games = team_games(frame)
    mine = games[(games["team"] == row["team"]) & (games["year"] == row["year"])]
    assert row["early"] == pytest.approx(mine.head(4)["residual"].mean())
    assert row["late"] == pytest.approx(mine.iloc[4:]["residual"].mean())
    assert row["n"] == len(mine)


def test_summary_splits_signal_from_noise() -> None:
    frame, _ = league(teams=60, seasons=8, rounds=14, tau=3.0, sigma=12.0)
    result = team_season_table(frame)
    summary = summarize(result.table, result.tau, result.sigma)
    # 12 / sqrt(14) = 3.2 of noise on a season's mean, 3 of planted signal.
    assert summary.noise_sd == pytest.approx(12 / np.sqrt(14), rel=0.1)
    assert summary.excess_sd == pytest.approx(3.0, abs=0.7)
    assert 0.35 < summary.reliability < 0.6
    # Effects are drawn fresh each season, so nothing carries over.
    assert abs(summary.year_to_year_corr) < 0.15


def test_null_counts_see_planted_effects_and_not_their_absence() -> None:
    planted, _ = league(teams=60, seasons=6, rounds=14, tau=4.0, sigma=12.0)
    flat, _ = league(teams=60, seasons=6, rounds=14, tau=0.0, sigma=12.0)
    for frame, real in ((planted, True), (flat, False)):
        result = team_season_table(frame)
        counts = null_counts(frame, result.table, result.sigma, draws=100)
        two = counts.set_index("threshold").loc[2.0]
        if real:
            assert two["observed"] > two["null_p95"]
        else:
            assert two["observed"] <= two["null_p95"] + 3


def test_null_counts_only_count_the_team_seasons_asked_about() -> None:
    frame, _ = league()
    result = team_season_table(frame)
    some = result.table[result.table["year"] == 2011]
    counts = null_counts(frame, some, result.sigma, draws=10)
    assert (counts["team_seasons"] == len(some)).all()


def test_a_group_s_level_comes_only_from_games_against_other_groups() -> None:
    # Two conferences; the first is 6 points worse than rated as a whole.
    effects = {(f"team{i:02d}", 2010): (-6.0 if i < 10 else 0.0) for i in range(20)}
    frame, _ = league(
        teams=20, seasons=1, rounds=19, tau=0.0, sigma=6.0, effects=effects
    )
    conference = {f"team{i:02d}": ("low" if i < 10 else "high") for i in range(20)}
    result = group_season_table(
        frame, frame["home_team"].map(conference), frame["away_team"].map(conference)
    )
    table = result.table.set_index("group")
    cross = (
        frame["home_team"].map(conference) != frame["away_team"].map(conference)
    ).sum()
    assert table.loc["low", "n"] == cross
    assert table.loc["low", "raw"] == pytest.approx(-6.0, abs=1.5)
    shrunk = table["shrunk"].to_dict()
    assert shrunk["low"] < shrunk["high"]


def test_select_matches_divisions_by_alias_and_case() -> None:
    frame = pd.DataFrame(
        {
            "team": ["a", "b", "c"],
            "year": [2020, 2021, 2021],
            "division": ["FBS", "NCAA Division II", "FBS"],
        }
    )
    assert select(frame, "fbs")["team"].tolist() == ["a", "c"]
    assert select(frame, "d2")["team"].tolist() == ["b"]
    assert select(frame, since=2021, until=2021)["team"].tolist() == ["b", "c"]
    with pytest.raises(ValueError, match="none of"):
        select(frame, "FCS")


def test_label_tiers_names_the_conference_without_the_division() -> None:
    frame = pd.DataFrame({"team": ["a", "b"], "year": [2020, 2020]})
    tiers = {("a", 2020): ("FBS", "Big Ten"), ("b", 2020): ("FBS", None)}
    labelled = label_tiers(frame, lambda team, year: tiers[(team, year)])
    assert labelled["conference"].tolist() == ["Big Ten", "FBS"]
    assert labelled["division"].tolist() == ["FBS", "FBS"]
    assert label_tiers(frame, None) is frame


def test_game_log_is_one_team_s_games() -> None:
    frame, _ = league(teams=4, seasons=2, rounds=3)
    log = game_log(team_games(frame), "team01", 2011)
    assert len(log) == 3
    assert (log["year"] == 2011).all()

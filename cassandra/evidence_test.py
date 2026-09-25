"""Tests for the idea-testing harness, on leagues with the answer planted."""

import numpy as np
import pandas as pd
import pytest

from .evidence import (
    buckets,
    cross_validated_gain,
    feature_effects,
    game_feature_effects,
    joint_fits,
    margin_scale,
    prepare_features,
)
from .residuals import MARGIN_RESIDUAL, PREDICTED_MARGIN
from .team_seasons import team_games
from .team_seasons_test import SCALE, league

PERMUTATIONS = 200


def _flagged(
    effect: float, share: float = 0.2, seed: int = 1, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A league where a random `share` of team-seasons are `effect` points off, and the flags."""
    rng = np.random.default_rng(seed)
    teams, seasons = kwargs.pop("teams", 40), kwargs.pop("seasons", 8)
    flags = {
        (f"team{i:02d}", 2010 + s): bool(rng.random() < share)
        for s in range(seasons)
        for i in range(teams)
    }
    effects = {key: effect if flagged else 0.0 for key, flagged in flags.items()}
    frame, _ = league(
        teams=teams, seasons=seasons, tau=0.0, effects=effects, seed=seed, **kwargs
    )
    features = pd.DataFrame(
        [{"team": t, "year": y, "flag": f} for (t, y), f in flags.items()]
    )
    return frame, features


def test_prepare_features_makes_numbers_and_one_hots_text() -> None:
    raw = pd.DataFrame(
        {
            "team": ["a", "b", "c"],
            "year": [2020, 2020, 2020],
            "flag": [True, False, True],
            "size": [1.5, None, 2.0],
            "reason": ["fired", None, "left"],
        }
    )
    features = prepare_features(raw)
    assert features["flag"].tolist() == [1.0, 0.0, 1.0]
    assert np.isnan(features["size"].iloc[1])
    assert features["reason=fired"].tolist() == [1.0, 0.0, 0.0]
    assert features["reason=left"].tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="more than one row"):
        prepare_features(pd.concat([raw, raw]))


def test_a_planted_effect_is_found_with_its_size() -> None:
    frame, raw = _flagged(-4.0, sigma=10.0)
    effects = feature_effects(
        team_games(frame), prepare_features(raw), permutations=PERMUTATIONS
    )
    whole = next(e for e in effects if e.window == "all games")
    assert whole.slope == pytest.approx(-4.0, abs=1.0)
    assert whole.t < -4
    assert whole.perm_p < 0.02
    assert whole.nonzero == raw["flag"].sum()


def test_nothing_planted_reads_as_nothing() -> None:
    frame, raw = _flagged(0.0, sigma=10.0)
    effects = feature_effects(
        team_games(frame), prepare_features(raw), permutations=PERMUTATIONS
    )
    whole = next(e for e in effects if e.window == "all games")
    assert abs(whole.t) < 3
    assert whole.perm_p > 0.01


def test_windows_split_a_team_s_season_by_game_number() -> None:
    frame, raw = _flagged(-4.0, sigma=10.0, rounds=10)
    effects = feature_effects(
        team_games(frame), prepare_features(raw), early_games=3, permutations=10
    )
    windows = {e.window: e for e in effects}
    assert set(windows) == {"games 1-3", "games 4+", "all games"}
    total = windows["games 1-3"].games + windows["games 4+"].games
    assert total == windows["all games"].games


def test_the_market_column_reads_whether_the_line_moved_with_the_feature() -> None:
    frame, raw = _flagged(-4.0, sigma=10.0)
    flagged = set(map(tuple, raw.loc[raw["flag"], ["team", "year"]].to_numpy()))
    # A market that knew: it sits 4 points under the model on every flagged side.
    home_flag = [(t, y) in flagged for t, y in zip(frame["home_team"], frame["year"])]
    away_flag = [(t, y) in flagged for t, y in zip(frame["away_team"], frame["year"])]
    frame["market_margin"] = (
        frame[PREDICTED_MARGIN] - 4.0 * np.array(home_flag) + 4.0 * np.array(away_flag)
    )
    effects = feature_effects(team_games(frame), prepare_features(raw), permutations=10)
    whole = next(e for e in effects if e.window == "all games")
    # -4 on a flagged side, +4 back whenever its opponent is flagged too --
    # which happens to flagged and unflagged teams alike, so only on average.
    assert whole.market_slope == pytest.approx(-4.0, abs=0.3)
    assert whole.n_lined == whole.games


def test_the_joint_fit_separates_two_features_that_travel_together() -> None:
    rng = np.random.default_rng(3)
    keys = [(f"team{i:02d}", 2010 + s) for s in range(8) for i in range(40)]
    first = rng.random(len(keys)) < 0.4
    # The second is mostly the first, and carries nothing of its own.
    second = np.where(rng.random(len(keys)) < 0.8, first, ~first)
    effects = {key: -4.0 * f for key, f in zip(keys, first)}
    frame, _ = league(teams=40, seasons=8, tau=0.0, sigma=10.0, effects=effects, seed=3)
    raw = pd.DataFrame(
        [
            {"team": t, "year": y, "real": a, "tagalong": b}
            for (t, y), a, b in zip(keys, first, second)
        ]
    )
    games, features = team_games(frame), prepare_features(raw)
    alone = {
        e.feature: e
        for e in feature_effects(games, features, permutations=10)
        if e.window == "all games"
    }
    assert alone["tagalong"].t < -3
    fit = next(f for f in joint_fits(games, features) if f.window == "all games")
    coefficients = fit.coefficients.set_index("feature")
    assert coefficients.loc["real", "slope"] == pytest.approx(-4.0, abs=1.2)
    assert abs(coefficients["t"].to_dict()["tagalong"]) < 3


def test_margin_scale_recovers_the_logistic() -> None:
    frame, _ = league(teams=6, seasons=1, rounds=4)
    assert margin_scale(frame) == pytest.approx(SCALE, rel=1e-6)


def test_the_cross_validated_shift_helps_when_the_effect_is_real() -> None:
    frame, raw = _flagged(-6.0, sigma=10.0)
    gain = cross_validated_gain(frame, team_games(frame), prepare_features(raw))
    assert gain.mae_change < -0.1
    assert gain.brier_change < 0
    assert gain.games == len(frame)
    assert 0 < gain.touched < gain.games


def test_the_cross_validated_shift_costs_a_little_when_it_isn_t() -> None:
    frame, raw = _flagged(0.0, sigma=10.0)
    gain = cross_validated_gain(frame, team_games(frame), prepare_features(raw))
    assert gain.mae_change > -0.05


def test_the_shift_goes_the_right_way_at_a_neutral_site() -> None:
    frame, raw = _flagged(-6.0, sigma=10.0)
    frame["neutral_site"] = True
    gain = cross_validated_gain(frame, team_games(frame), prepare_features(raw))
    assert gain.mae_change < -0.1


def test_a_game_feature_is_read_from_the_home_side() -> None:
    frame, _ = league(teams=30, seasons=6, tau=0.0, sigma=10.0, seed=4)
    rng = np.random.default_rng(4)
    rest = rng.integers(-3, 4, size=len(frame)).astype(float)
    frame[MARGIN_RESIDUAL] = frame[MARGIN_RESIDUAL] + 1.5 * rest
    features = pd.DataFrame({"game_id": frame["game_id"], "rest": rest})
    (effect,) = game_feature_effects(frame, features, permutations=PERMUTATIONS)
    assert effect.slope == pytest.approx(1.5, abs=0.3)
    assert effect.perm_p < 0.02


def test_buckets_show_where_a_relationship_lives() -> None:
    x = pd.Series(np.arange(100, dtype=float))
    residual = pd.Series(np.where(x >= 80, 5.0, 0.0))
    table = buckets(x, residual)
    assert len(table) == 5
    assert table["mean"].iloc[-1] == pytest.approx(5.0)
    assert (table["mean"].iloc[:-1] == 0).all()
    edged = buckets(x, residual, edges=[50])
    assert edged["n"].tolist() == [51, 49]

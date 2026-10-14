import numpy as np
import pandas as pd
import pytest
from collide import GRID, League, Params, collide
from scipy import stats
from scipy.special import logit


def arr(*v):
    return np.array(v, dtype=float)


def test_a_known_average_defense_is_the_conjugate_update():
    """Defense pinned at the league rate, binomial game: Beta(a + k, b + n - k)."""
    league, a, b, n, k = 0.45, 130.5, 159.5, 63, 34
    pinned = (arr(league * 1e7), arr((1 - league) * 1e7))
    out = collide((arr(a), arr(b)), pinned, logit(league), arr(0), arr(n), arr(k), np.inf, size=96)
    assert out.off[0][0] == pytest.approx(a + k, rel=2e-3)
    assert out.off[1][0] == pytest.approx(b + n - k, rel=2e-3)
    assert out.loglik[0] == pytest.approx(stats.betabinom.logpmf(k, n, a, b), abs=1e-3)


def test_offense_and_defense_are_symmetric():
    o, d = (arr(140.0), arr(150.0)), (arr(260.0), arr(330.0))
    there = collide(o, d, logit(0.45), arr(0), arr(60), arr(31), 200.0)
    back = collide(d, o, logit(0.45), arr(0), arr(60), arr(31), 200.0)
    np.testing.assert_allclose(there.off, back.deff, rtol=1e-9)
    np.testing.assert_allclose(there.p, back.p, rtol=1e-12)


def test_the_less_certain_side_takes_the_surprise():
    loose = (arr(0.45 * 100), arr(0.55 * 100))
    tight = (arr(0.45 * 2000), arr(0.55 * 2000))
    out = collide(loose, tight, logit(0.45), arr(0), arr(60), arr(40), np.inf)
    moved_off = out.off[0] / (out.off[0] + out.off[1]) - 0.45
    moved_def = out.deff[0] / (out.deff[0] + out.deff[1]) - 0.45
    assert moved_off[0] > 5 * moved_def[0] > 0


def test_an_uncertain_matchup_spreads_the_prediction():
    """Same means, wider betas: a fatter predictive, so an extreme game is likelier."""
    sure = collide((arr(4500.0), arr(5500.0)), (arr(4500.0), arr(5500.0)),
                   logit(0.45), arr(0), arr(60), arr(45), np.inf)
    unsure = collide((arr(45.0), arr(55.0)), (arr(45.0), arr(55.0)),
                     logit(0.45), arr(0), arr(60), arr(45), np.inf)
    assert unsure.loglik[0] > sure.loglik[0]


def test_recentring_leaves_every_matchup_where_it_was():
    league = League(["a", "b", "c"], Params(), rate=0.45)
    league.x_off[:] = [0.2, 0.1, 0.0]
    league.x_def[:] = [-0.1, 0.05, 0.2]
    before = league.lam + league.x_off[:, None] + league.x_def[None, :] - league.lam
    lam_before = league.lam
    league.recentre()
    after = league.x_off[:, None] + league.x_def[None, :]
    np.testing.assert_allclose(before, after + (league.lam - lam_before))
    assert league.x_off.mean() == pytest.approx(0) and league.x_def.mean() == pytest.approx(0)


def test_the_offseason_regresses_and_widens():
    params = Params(kappa_off=300, carry_off=0.5)
    league = League(["a", "b"], params, rate=0.45)
    league.x_off[:] = [0.3, -0.3]
    league.k_off[:] = 2000
    league.new_season()
    np.testing.assert_allclose(league.x_off, [0.15, -0.15])
    assert (league.k_off < 2000).all() and (league.k_off > 300).all()


def test_a_week_of_drift_widens_each_side_by_its_own_amount_and_keeps_the_mean():
    league = League(["a", "b"], Params(drift_off=0.01, drift_def=0.0), rate=0.45)
    league.x_off[:] = [0.2, -0.2]
    league.k_off[:] = 1500
    league.k_def[:] = 1500
    before = league.betas("off", np.arange(2))
    league.age()
    after = league.betas("off", np.arange(2))
    np.testing.assert_allclose(before[0] / (before[0] + before[1]), after[0] / (after[0] + after[1]))
    var = lambda a, b: a * b / ((a + b) ** 2 * (a + b + 1))  # noqa: E731
    np.testing.assert_allclose(var(*after) - var(*before), 0.01**2, rtol=1e-6)
    assert (league.k_def == 1500).all()


def test_drift_round_trips_through_the_search_vector():
    params = Params(drift_off=0.004, drift_def=0.0)
    back = Params.from_vector(params.to_vector())
    assert back.drift_off == pytest.approx(0.004)
    assert back.drift_def == pytest.approx(0.0, abs=1e-12)


def test_a_week_updates_both_sides_of_both_meetings():
    league = League(["a", "b"], Params(), rate=0.45)
    week = pd.DataFrame(
        {"offense": ["a", "b"], "defense": ["b", "a"], "home": [1, -1],
         "n": [60, 60], "k": [40, 20]}
    )
    out = league.play_week(week, size=GRID)
    assert out.p.iloc[0] > out.p.iloc[1]  # home edge
    assert league.x_off[0] > 0 > league.x_off[1]
    assert league.x_def[1] > 0 > league.x_def[0]

"""Tests for the residual diagnostic.

The frames here are built two ways on purpose. `add_residuals` is tested
against a real prob->margin fit, because the fit is the part of it worth
checking; everything downstream is handed a frame with `margin_residual`
already in it, so a planted effect can be planted exactly rather than
inferred through a logistic. A test whose expected sigma depended on what a
fitter did to synthetic win probabilities would be testing the fitter.
"""

import numpy as np
import pandas as pd
import pytest

from .residuals import (
    MARGIN_RESIDUAL,
    MARKET_MARGIN,
    PREDICTED_MARGIN,
    UNCLASSIFIED,
    add_residuals,
    axis_report,
    classification_axes,
    favorite_size,
    home_field_report,
    home_field_table,
    rest_advantage,
    season_stage,
    standard_axes,
)

# Enough for a null's mean and sd to be steady between runs without every
# test paying for the default 400.
PERMUTATIONS = 200


def _frame(
    residuals: np.ndarray,
    home_teams: list[str] | None = None,
    away_teams: list[str] | None = None,
    weeks: np.ndarray | None = None,
    market_margin: np.ndarray | None = None,
    neutral: np.ndarray | None = None,
) -> pd.DataFrame:
    """A post-`add_residuals` frame with the residual planted directly.

    The columns the reports read, and nothing else: what a real frame carries
    beyond these is `save_predictions`' business, and a diagnostic that
    happened to depend on one of them would pass here and fail there.
    """
    n = len(residuals)
    # Actual margin is arbitrary given the residual; predicted is whatever
    # makes the two consistent, so `market_gap` and `favorite_size` have real
    # numbers to work from.
    predicted = np.linspace(-14, 14, n)
    mov = predicted + residuals
    return pd.DataFrame(
        {
            "team1_mov": mov,
            "team1_win_prob": np.clip(0.5 + predicted / 60, 0.02, 0.98),
            "team1_win": mov > 0,
            PREDICTED_MARGIN: predicted,
            MARGIN_RESIDUAL: residuals,
            MARKET_MARGIN: (
                np.full(n, np.nan) if market_margin is None else market_margin
            ),
            "week_number": np.arange(n) % 15 + 1 if weeks is None else weeks,
            "year": 2020,
            "home_team": [f"h{i % 8}" for i in range(n)]
            if home_teams is None
            else home_teams,
            "away_team": [f"a{i % 8}" for i in range(n)]
            if away_teams is None
            else away_teams,
            "neutral_site": np.zeros(n, dtype=bool) if neutral is None else neutral,
            # A game a week, which makes every rest advantage zero unless a
            # test moves one.
            "date": pd.Timestamp("2020-09-05")
            + pd.to_timedelta((np.arange(n) % 15) * 7, unit="D"),
        }
    )


def test_add_residuals_is_actual_minus_implied():
    games = pd.DataFrame(
        {
            "home_score": [24, 10, 14, 31],
            "away_score": [20, 30, 7, 3],
            "team1_win_prob": [0.7, 0.4, 0.65, 0.85],
            "team1_win": [True, False, True, True],
            "spread": [-3.0, 6.0, None, -10.0],
        }
    )
    scored = add_residuals(games)
    assert (scored["team1_mov"] == [4, -20, 7, 28]).all()
    # The residual is defined off the fit, whatever the fit came out at.
    np.testing.assert_allclose(
        scored[MARGIN_RESIDUAL],
        scored["team1_mov"] - scored[PREDICTED_MARGIN],
    )
    # A spread is negated to get the margin it implies, and a game with no
    # line carries no market margin rather than a zero.
    np.testing.assert_allclose(scored[MARKET_MARGIN], [3.0, -6.0, np.nan, 10.0])


def test_add_residuals_survives_a_league_with_no_lines():
    """nfl, today: every game's spread is None, so the column is `object`.

    The four-row frame above coerces its one None to NaN because the other
    three are floats, which is why this needs its own test -- an all-missing
    column is a different dtype and used to raise on the negation.
    """
    games = pd.DataFrame(
        {
            "home_score": [24, 10, 14, 31],
            "away_score": [20, 30, 7, 3],
            "team1_win_prob": [0.7, 0.4, 0.65, 0.85],
            "team1_win": [True, False, True, True],
            "spread": [None, None, None, None],
        }
    )
    scored = add_residuals(games)
    assert scored[MARKET_MARGIN].isna().all()
    # And the reports downstream say "no lines" rather than falling over.
    stats = axis_report(
        scored.assign(**{MARGIN_RESIDUAL: scored[MARGIN_RESIDUAL]}),
        pd.Series(["a", "a", "b", "b"], index=scored.index),
        "half",
        permutations=10,
        min_games=1,
    )
    assert all(np.isnan(s.market_gap) and s.n_lined == 0 for s in stats.slices)


def test_add_residuals_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="No games"):
        add_residuals(pd.DataFrame())


def test_a_flat_axis_reads_as_flat():
    rng = np.random.default_rng(0)
    # Residuals with no relationship to the week they landed in, which is
    # what an axis carrying nothing looks like.
    df = _frame(rng.normal(0, 14, 6000))
    report = axis_report(
        df, df["week_number"].astype(str), "week", permutations=PERMUTATIONS
    )
    assert len(report.slices) == 15
    # Two claims, and the second is the one that matters: not just that the
    # per-slice biases are small, but that they are no larger than shuffling
    # the same residuals produces.
    assert abs(report.sigma) < 3
    # Not zero: `signal_points` floors a noisy difference of two noisy
    # numbers, so a flat axis still reports a fraction of a point. Across
    # seeds this lands under 0.6 on 6,000 games in fifteen slices, and the
    # bound is what says the floor is small rather than absent.
    assert report.signal_points < 0.8


def test_a_planted_week_effect_separates_from_its_null():
    rng = np.random.default_rng(1)
    n = 6000
    weeks = np.arange(n) % 15 + 1
    # The shape the recruiting-and-transfers question predicts: the model
    # runs three points light on home teams early and is unbiased later.
    planted = np.where(weeks <= 4, 3.0, 0.0)
    df = _frame(rng.normal(0, 14, n) + planted, weeks=weeks)
    report = axis_report(
        df, df["week_number"].astype(str), "week", permutations=PERMUTATIONS
    )
    assert report.sigma > 4
    # 4 of 15 weeks sit 3 points above the other 11, so about the grand mean
    # of 0.8 the deviations are +2.2 and -0.8, an RMS of
    # sqrt(4/15 * 2.2^2 + 11/15 * 0.8^2) ~ 1.33.
    assert 1.0 < report.signal_points < 1.7
    # The per-slice bias stays uncentered, because "the model is 2.7 points
    # light in week 1" is the sentence a table row is for. Averaged over the
    # four weeks rather than checked one at a time: a single week is 400
    # games at a standard error of 0.7, so one of them landing at 2.1 is the
    # sample, not the code.
    early = [s.margin_bias for s in report.slices if int(s.label) <= 4]
    late = [s.margin_bias for s in report.slices if int(s.label) > 4]
    assert np.mean(early) == pytest.approx(3.0, abs=0.5)
    assert np.mean(late) == pytest.approx(0.0, abs=0.5)
    assert report.overall_bias == pytest.approx(0.8, abs=0.4)


def test_signal_points_shrinks_with_the_effect_and_sigma_does_not_have_to():
    """The two columns answer different questions, and the test says so.

    A tenth of a point of real bias on a big enough sample is many sigma away
    from its null and still worth nothing. Reading sigma alone is the failure
    mode this pair of columns exists to prevent.
    """
    rng = np.random.default_rng(2)
    n = 150000
    weeks = np.arange(n) % 15 + 1
    df = _frame(rng.normal(0, 14, n) + np.where(weeks <= 4, 0.4, 0.0), weeks=weeks)
    report = axis_report(
        df, df["week_number"].astype(str), "week", permutations=PERMUTATIONS
    )
    assert report.sigma > 3
    assert report.signal_points < 0.3


def test_small_slices_leave_the_table_and_the_null_together():
    rng = np.random.default_rng(3)
    labels = ["big"] * 500 + ["also_big"] * 500 + ["tiny"] * 3
    df = _frame(rng.normal(0, 14, len(labels)))
    report = axis_report(
        df,
        pd.Series(labels, index=df.index),
        "sizes",
        min_games=20,
        permutations=PERMUTATIONS,
    )
    assert {s.label for s in report.slices} == {"big", "also_big"}
    # The three dropped games are out of the null as well, so the sigma is
    # computed over the same games the table reports.
    assert sum(s.n for s in report.slices) == 1000


def test_one_slice_has_nothing_to_disagree_with():
    df = _frame(np.zeros(50))
    report = axis_report(
        df, pd.Series(["only"] * 50, index=df.index), "single", permutations=10
    )
    assert report.sigma == 0.0
    assert report.signal_points == 0.0


def test_market_gap_is_the_model_minus_the_market():
    # The model is 4 points off every game; the market is 1 point off.
    n = 100
    residuals = np.full(n, 4.0)
    df = _frame(residuals)
    df = df.assign(**{MARKET_MARGIN: df["team1_mov"] - 1.0})
    report = axis_report(
        df, pd.Series(["all"] * n, index=df.index), "one", permutations=10
    )
    (only,) = report.slices
    assert only.n_lined == n
    # Positive means the market won, by the 3 points that separate them.
    assert only.market_gap == pytest.approx(3.0)


def test_market_gap_is_nan_without_lines():
    df = _frame(np.full(50, 1.0))
    report = axis_report(
        df, pd.Series(["all"] * 50, index=df.index), "one", permutations=10
    )
    (only,) = report.slices
    assert only.n_lined == 0
    assert np.isnan(only.market_gap)


def _home_and_away(
    rng: np.random.Generator,
    teams: list[str],
    games_per_pair: int,
    home_edge: dict[str, float],
    rating_error: dict[str, float] | None = None,
    noise: float = 14.0,
) -> pd.DataFrame:
    """A round robin, home and away, with per-team effects planted in.

    `home_edge[t]` is the points of home advantage team t has *beyond* the
    league constant the model applied. It lands in the residual the way a
    converged replay leaves it rather than the way it happens: half of it has
    already been absorbed into t's rating, since t plays half its schedule at
    home, so what is left is +d/2 whenever t is home and +d/2 (from the home
    side's view) whenever t is away. Planting the raw +d on home games only
    would be planting a rating error and a home edge at once, and then
    testing that the two columns separate would be testing nothing.

    `rating_error[t]` is a team the model is simply that many points light
    on, everywhere: +e when t is home and -e when t is away, from the home
    team's side.
    """
    rating_error = rating_error or {}
    homes, aways, residuals = [], [], []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            for _ in range(games_per_pair):
                homes.append(home)
                aways.append(away)
                residuals.append(
                    (home_edge.get(home, 0.0) + home_edge.get(away, 0.0)) / 2
                    + rating_error.get(home, 0.0)
                    - rating_error.get(away, 0.0)
                    + rng.normal(0, noise)
                )
    return _frame(
        np.array(residuals),
        home_teams=homes,
        away_teams=aways,
        weeks=np.arange(len(homes)) % 15 + 1,
    )


def test_home_field_splits_a_rating_error_from_a_home_edge():
    rng = np.random.default_rng(4)
    teams = [f"t{i}" for i in range(10)]
    # t0 has a real home edge; t1 is simply under-rated everywhere. Only the
    # first should show up as home_excess, and only the second as
    # rating_error -- that separation is the reason the table has two columns.
    df = _home_and_away(rng, teams, 40, home_edge={"t0": 6.0}, rating_error={"t1": 6.0})
    table = {t.team: t for t in home_field_table(df)}
    assert table["t0"].home_excess == pytest.approx(6.0, abs=1.5)
    assert table["t0"].rating_error == pytest.approx(0.0, abs=1.5)
    assert table["t1"].rating_error == pytest.approx(6.0, abs=1.5)
    assert table["t1"].home_excess == pytest.approx(0.0, abs=1.5)


def test_home_field_is_sorted_by_the_excess():
    rng = np.random.default_rng(5)
    df = _home_and_away(rng, [f"t{i}" for i in range(6)], 40, {"t3": 12.0})
    table = home_field_table(df)
    assert table[0].team == "t3"
    assert [t.home_excess for t in table] == sorted(
        (t.home_excess for t in table), reverse=True
    )


def test_home_field_leaves_neutral_games_out():
    rng = np.random.default_rng(6)
    df = _home_and_away(rng, [f"t{i}" for i in range(6)], 40, {})
    # Every one of t0's home games moved to a neutral site, so it drops below
    # the threshold on the home side and out of the table entirely.
    df = df.assign(neutral_site=(df["home_team"] == "t0"))
    assert "t0" not in {t.team for t in home_field_table(df, min_games=20)}
    assert "t1" in {t.team for t in home_field_table(df, min_games=20)}


def test_home_field_needs_both_halves_of_the_difference():
    rng = np.random.default_rng(7)
    df = _home_and_away(rng, [f"t{i}" for i in range(6)], 40, {})
    # A team with plenty of home games and three away ones has no usable
    # difference, however good its home sample is.
    keep = ~((df["away_team"] == "t0") & (np.arange(len(df)) % 20 != 0))
    assert "t0" not in {t.team for t in home_field_table(df[keep], min_games=20)}


def test_home_field_wants_a_neutral_site_column():
    df = _home_and_away(np.random.default_rng(8), ["a", "b", "c"], 40, {}).drop(
        columns=["neutral_site"]
    )
    with pytest.raises(ValueError, match="neutral_site"):
        home_field_table(df)


def test_a_league_with_one_home_advantage_reads_as_flat():
    rng = np.random.default_rng(9)
    # Every team's home edge is the league constant, which the model already
    # applied -- so what's left is noise, and the report should say so rather
    # than reporting the spread of ten noisy differences.
    df = _home_and_away(rng, [f"t{i}" for i in range(12)], 30, {})
    report = home_field_report(df, permutations=PERMUTATIONS)
    assert abs(report.sigma) < 3
    # A full point of noise floor, on twelve teams with 330 games each side.
    # Worth reading twice before running this on a real league: a per-team
    # home edge is a difference of two noisy means, and a table of them
    # always has a top row that looks like a finding.
    assert report.signal_points < 1.5


def test_a_real_spread_of_home_edges_separates():
    rng = np.random.default_rng(10)
    teams = [f"t{i}" for i in range(12)]
    # Half the league is worth 5 points more at home than the constant it was
    # given, half is worth 5 less: an sd of 5 around the league number.
    edges = {t: (5.0 if i % 2 else -5.0) for i, t in enumerate(teams)}
    df = _home_and_away(rng, teams, 30, edges)
    report = home_field_report(df, permutations=PERMUTATIONS)
    assert report.sigma > 10
    # signal_points estimates the sd of the per-team edge, which is 5 here.
    assert 3.5 < report.signal_points < 6.5


def test_season_stage_cuts_where_it_says_it_does():
    df = pd.DataFrame({"week_number": [1, 4, 5, 11, 12, 16]})
    assert list(season_stage(df)) == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]


def test_favorite_size_keeps_the_sign():
    df = pd.DataFrame({PREDICTED_MARGIN: [-20.0, -5.0, 5.0, 20.0, 0.5]})
    buckets = favorite_size(df)
    # A 20-point home favorite and a 20-point road favorite are different
    # slices; folding the sign away would hide a home advantage error.
    assert buckets.iloc[0] != buckets.iloc[3]
    assert buckets.iloc[1] != buckets.iloc[2]
    # A near-pick'em is one bucket either side of level, not two: the
    # innermost edge straddles zero on purpose, since splitting a coin flip
    # by which way it leaned is splitting noise.
    assert buckets.iloc[4] == "(-3.0, 3.0]"


def test_standard_axes_all_run():
    rng = np.random.default_rng(11)
    df = _frame(rng.normal(0, 14, 2000))
    for name, labels in standard_axes(df).items():
        report = axis_report(df, labels, name, permutations=20)
        assert report.axis == name
        assert len(labels) == len(df)


def _rest_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Games as (home, away, date), in the columns `rest_advantage` reads."""
    return pd.DataFrame(
        {
            "home_team": [r[0] for r in rows],
            "away_team": [r[1] for r in rows],
            "date": pd.to_datetime([r[2] for r in rows]),
        }
    )


def test_rest_advantage_counts_from_each_team_s_own_last_game():
    # a and b both play on the 1st. a is back on the 8th (7 days) while b
    # played again on the 5th and so comes in on 3 -- a four-day edge to the
    # home side.
    df = _rest_frame(
        [
            ("a", "b", "2020-09-01"),
            ("b", "c", "2020-09-05"),
            ("a", "b", "2020-09-08"),
        ]
    )
    buckets = rest_advantage(df)
    # Neither team in the first game has a previous one, and c has none in
    # the second.
    assert buckets.iloc[0] == "unknown"
    assert buckets.iloc[1] == "unknown"
    assert buckets.iloc[2] == "(2.0, 5.0]"


def test_rest_advantage_is_signed_toward_the_home_team():
    df = _rest_frame(
        [
            ("a", "b", "2020-09-01"),
            ("a", "c", "2020-09-05"),
            ("b", "a", "2020-09-08"),
        ]
    )
    # b has been off since the 1st (7 days); a played on the 5th (3). b is
    # home, so the advantage is positive.
    assert rest_advantage(df).iloc[2] == "(2.0, 5.0]"


def test_rest_advantage_leaves_the_offseason_out():
    df = _rest_frame(
        [
            ("a", "b", "2020-11-01"),
            ("a", "b", "2021-09-04"),
        ]
    )
    # Ten months is not a bye week, and a rest table with a 307-day row in it
    # is a table that gets misread.
    assert rest_advantage(df).iloc[1] == "unknown"


def test_rest_advantage_survives_a_duplicated_index():
    df = _rest_frame(
        [
            ("a", "b", "2020-09-01"),
            ("b", "c", "2020-09-05"),
            ("a", "b", "2020-09-08"),
        ]
    )
    duplicated = df.set_index(pd.Index([0, 0, 0]))
    assert list(rest_advantage(duplicated)) == list(rest_advantage(df))


def test_classification_axes_are_empty_for_an_unclassified_league():
    """nfl: `call_it_what_you_want` files no divisions for it.

    Empty rather than a frame of "unclassified", so a caller merging these
    into `standard_axes` gets no axis at all rather than one that reports
    zeros -- the two look identical in a printed table.
    """
    df = pd.DataFrame(
        {
            "year": [2020, 2020],
            "home_team": ["packers", "bears"],
            "away_team": ["bears", "packers"],
        }
    )
    assert classification_axes(df, "nfl") == {}


def test_classification_axes_label_by_division_and_matchup():
    """A real ncaafb frame, against the classifications that ship with ciwyw.

    Pinned to teams whose tier hasn't moved in the seasons named, and
    asserted on the *shape* of the labels rather than on the exact division
    strings, which are ESPN's and are the package's to rename.
    """
    df = pd.DataFrame(
        {
            "year": [2015, 2015],
            "home_team": ["Alabama Crimson Tide", "Alabama Crimson Tide"],
            "away_team": ["Alabama Crimson Tide", "not a team at all"],
        }
    )
    axes = classification_axes(df, "ncaafb")
    assert set(axes) == {"division", "conference", "division_matchup"}
    division = axes["division"]
    # Both rows are the same home team, so both carry the same division, and
    # it is a real one rather than the fallback.
    assert division.iloc[0] == division.iloc[1] != UNCLASSIFIED
    # A conference label carries its division too, so two conferences with
    # the same name in different tiers stay apart.
    assert axes["conference"].iloc[0].startswith(division.iloc[0])
    # The matchup is directional and names both sides...
    assert axes["division_matchup"].iloc[0] == (
        f"{division.iloc[0]} at home vs {division.iloc[0]}"
    )
    # ...and a game with an unplaceable opponent gets no matchup at all,
    # rather than one half of a label.
    assert axes["division_matchup"].iloc[1] == UNCLASSIFIED
    assert axes["division"].iloc[1] != UNCLASSIFIED


def test_an_independent_falls_back_to_its_division():
    """A team with no conference gets its division, not a shared None bucket.

    Built by hand rather than found in the registry: which schools are
    independent moves season to season, and a test that pinned one would
    start failing on a data update that isn't about this code.
    """
    df = pd.DataFrame(
        {
            "year": [2015],
            "home_team": ["Notre Dame Fighting Irish"],
            "away_team": ["Notre Dame Fighting Irish"],
        }
    )
    axes = classification_axes(df, "ncaafb")
    conference = axes["conference"].iloc[0]
    division = axes["division"].iloc[0]
    assert conference == division or conference.startswith(f"{division} / ")

import re
from typing import cast

import pytest

import optimize
from cassandra.predictor import Predictor, frame
from cassandra.save_predictions import OddsDatabase
from optimize import _pinned_notice

# The pattern `.claude/skills/run-report/summarize_run.py` matches to pull a
# line out of a container log and into the report's tuning diagnostics. Copied
# rather than imported: the skill isn't an importable package, and the point of
# the test is that the two stay in step even so.
_REPORT_DIAGNOSTIC = re.compile(r"^\[optimize\] (?P<message>.+)$")


def test_pinned_notice__none_when_nothing_is_pinned():
    assert _pinned_notice({}, "elo.json") is None


def test_pinned_notice__names_every_pin_and_the_config_to_edit():
    notice = _pinned_notice(
        {"scoring_method": "sigmoid", "season_regression": 0}, "glicko_full.json"
    )

    assert notice is not None
    assert "season_regression=0" in notice
    assert "scoring_method=sigmoid" in notice
    # Without the filename the reader has a question and no way to act on it.
    assert "glicko_full.json" in notice


def test_pinned_notice__is_picked_up_by_the_run_report():
    """The whole mechanism: a pin has to survive into the weekly report.

    A pinned parameter emits no probes and so no bound-hit diagnostic, so this
    line is the only thing that carries the decision forward. If the prefix
    drifts from what `summarize_run.py` matches, the reminder disappears
    silently -- which is the failure this is here to prevent.
    """
    notice = _pinned_notice({"season_regression": 0}, "elo.json")

    assert notice is not None
    assert _REPORT_DIAGNOSTIC.match(notice)


def test_a_probe_reaches_the_constructor_in_its_own_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search moves knobs; the predictor takes arguments.

    `_score_probe` is the one place the two meet, so a framed config's probe
    -- points, a deviation budget -- has to arrive at the constructor as the
    rating-unit arguments it derives, with the pins merged in the same way.
    """
    seen: dict[str, object] = {}

    class Fake:
        def __init__(self, league: str, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(
        optimize,
        "join_with_odds",
        lambda predictor, seasons, odds_db, post_callbacks: [],
    )

    value = optimize._score_probe(
        "nfl",
        [],
        cast(OddsDatabase, None),
        cast(type[Predictor], Fake),
        lambda df: 1.0,
        frame=frame.POINTS,
        weeks_per_season=20,
        fixed={"scoring_method": "sigmoid", "travel_pts": 0},
        sigmoid_scale=10.0,
        hfa_pts=3.0,
    )

    assert value == 1.0
    assert seen["home_advantage"] == pytest.approx(
        3 * frame.rating_units_per_point(10.0)
    )
    assert seen["travel_advantage"] == 0.0
    assert seen["scoring_method"] == "sigmoid"
    assert "hfa_pts" not in seen and "travel_pts" not in seen

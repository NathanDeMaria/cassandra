import re

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

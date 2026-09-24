"""Reading say-youll-remember-me's rows into offseason facts."""

import pytest

from .offseason import OffseasonFacts, QuarterbackRow, quarterback_facts


def _qb(team, season, key, starts, week_one, attempts, epa) -> QuarterbackRow:
    return QuarterbackRow(team, season, key, starts, week_one, attempts, epa)


# Two FBS teams, two seasons. Averages: 2019 is 0.1 EPA per attempt.
ROWS = [
    _qb("X", 2019, "old x", 12, True, 400, 40.0),  # X's starter: average
    _qb("Y", 2019, "star y", 12, True, 400, 80.0),  # Y's: 0.2 per attempt
    _qb("Y", 2019, "backup y", 0, False, 60, 0.0),
    _qb("X", 2018, "old x", 12, True, 400, 40.0),
    _qb("Y", 2018, "star y", 12, True, 400, 40.0),
    # 2020: X gets Y's star by transfer; Y's backup takes over; a team
    # with no 2019 starter on file has nothing to compare.
    _qb("X", 2020, "star y", 12, True, 400, 60.0),
    _qb("Y", 2020, "backup y", 12, True, 400, 40.0),
    _qb("Z", 2020, "someone", 12, True, 400, 40.0),
]
FBS = {(t, y) for t in ("X", "Y", "Z") for y in (2018, 2019, 2020)}


def test_a_transfer_brings_his_record() -> None:
    new, change = quarterback_facts(ROWS, FBS)[("X", 2020)]
    assert new
    assert change > 0.05  # a 0.2 passer, shrunk, replacing an average one


def test_a_backup_promoted_is_worse_than_the_star_who_left() -> None:
    new, change = quarterback_facts(ROWS, FBS)[("Y", 2020)]
    assert new and change < -0.05


def test_a_returning_starter_is_no_change() -> None:
    assert quarterback_facts(ROWS, FBS)[("X", 2019)] == (False, 0.0)


def test_nothing_to_compare_is_no_fact() -> None:
    assert ("Z", 2020) not in quarterback_facts(ROWS, FBS)


def test_a_league_without_data_has_no_facts() -> None:
    facts = OffseasonFacts.for_league("mens")
    assert len(facts) == 0
    assert list(facts.seasons(2024)) == []


def test_the_league_facts_read_the_bundled_data() -> None:
    facts = OffseasonFacts.for_league("ncaafb")
    # Drew Mestemaker, North Texas to Oklahoma State.
    arrived = facts.get("Oklahoma State Cowboys", 2026)
    left = facts.get("North Texas Mean Green", 2026)
    assert arrived is not None and arrived.new_quarterback
    assert arrived.quality_change > 0.1
    assert left is not None and left.coach_departure == "left_for_job"
    assert left.quality_change < -0.1
    assert facts.get("Oklahoma State Cowboys", None) is None
    assert facts.get("LSU Tigers", 2019) == pytest.approx((None, False, 0.0))


def test_the_nfl_facts_are_keyed_by_nickname() -> None:
    facts = OffseasonFacts.for_league("nfl")
    assert facts.get("Las Vegas Raiders", 2022) is None
    raiders = facts.get("raiders", 2022)
    assert raiders is not None and raiders.coach_departure == "resigned"  # Gruden
    dolphins = facts.get("dolphins", 2007)
    assert dolphins is not None and dolphins.coach_departure == "left_for_job"


def test_nfl_quarterback_changes() -> None:
    facts = OffseasonFacts.for_league("nfl")
    # Stafford for Goff, both ways.
    rams, lions = facts.get("rams", 2021), facts.get("lions", 2021)
    assert rams is not None and rams.new_quarterback and rams.quality_change > 0
    assert lions is not None and lions.new_quarterback and lions.quality_change < 0
    chiefs = facts.get("chiefs", 2025)
    assert chiefs is not None and chiefs.new_quarterback is False

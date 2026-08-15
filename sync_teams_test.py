from dataclasses import replace

import pytest
from call_it_what_you_want import (
    ESPN,
    NCAA,
    TeamName,
    default_teams,
    local_csv,
    local_path,
)

from cassandra.alias_candidates import NameCandidate, TeamActivity
from cassandra.aliases import TeamNamer
from sync_teams import (
    Decisions,
    _duplicate_ids,
    _observations,
    _placeholder,
    _record_all,
    _record_same_as,
    _review_duplicate_ids,
    _review_names,
    _unrecorded,
)


@pytest.fixture(autouse=True)
def staging(tmp_path, monkeypatch):
    """Point the package's local records at a scratch directory.

    Everything here goes through `record`, which writes to the package's
    staging file -- so the tests need that file to be a temporary one rather
    than the developer's real registry.
    """
    monkeypatch.setenv("CIWYW_DATA_DIR", str(tmp_path))
    default_teams.cache_clear()
    yield tmp_path
    default_teams.cache_clear()


def _name(name: str, year: int, league: str = "ncaafb") -> TeamName:
    return TeamName(name, year, ESPN, league)


def test_placeholder_ids_are_dropped() -> None:
    """ESPN's "TBD" for an opponent that isn't announced yet."""
    assert _placeholder("-1")
    assert _placeholder("-2")
    assert not _placeholder("2341")


def test_observations_carry_the_league() -> None:
    payload = {
        "events": [
            {
                "competitions": [
                    {
                        "competitors": [
                            {"team": {"id": "150", "displayName": "Duke Blue Devils"}},
                            {"team": {"id": "-1", "displayName": "TBD"}},
                        ]
                    }
                ]
            }
        ]
    }
    ((espn_id, name),) = list(_observations(payload, 2025, "ncaafb"))
    assert espn_id == "150"
    assert name == _name("Duke Blue Devils", 2025)


def test_unrecorded_separates_new_teams_from_new_names() -> None:
    """Army (349) is in the package's bundled data; 900001 is nobody."""
    unrecorded = _unrecorded(
        default_teams(NCAA),
        {
            # Already on file. An observation has to match on league too, so
            # this is the ncaafb-tagged row the bundled data actually holds.
            ("349", _name("Army Black Knights", 2012)),
            ("349", _name("Army Cadets", 2005)),  # new name, known team
            ("900001", _name("Somewhere State", 2020)),  # new team
        },
    )

    assert [o[1].name for o in unrecorded.observations] == [
        "Army Cadets",
        "Somewhere State",
    ]
    assert unrecorded.new_ids == ["900001"]


def test_recording_stages_rows_and_takes_effect() -> None:
    _record_all([("349", _name("Army Cadets", 2005))], NCAA)

    # Written where the package stages things, not into cassandra.
    assert local_path(NCAA).is_file()
    assert "Army Cadets" in local_csv(NCAA)
    # And usable immediately, without a reinstall.
    assert default_teams(NCAA).current_name("Army Cadets", league="ncaafb")


def test_recording_the_same_observation_twice_stages_one_row() -> None:
    """A re-pull shouldn't grow the staging file."""
    _record_all([("349", _name("Army Cadets", 2005))], NCAA)
    before = local_csv(NCAA)
    _record_all([("349", _name("Army Cadets", 2005))], NCAA)
    assert local_csv(NCAA) == before


def test_same_as_pools_two_espn_records() -> None:
    """ESPN files one school under two ids; `same_as` is how that's said."""
    _record_all(
        [
            ("900001", _name("Bay State Otters", 2024)),
            ("900002", _name("Bay State Otters", 2025)),
        ],
        NCAA,
    )
    teams = default_teams(NCAA)
    assert teams.by_espn_id("900001") != teams.by_espn_id("900002")

    _record_same_as(NCAA, teams.by_espn_id("900001"), teams.by_espn_id("900002"))

    default_teams.cache_clear()
    teams = default_teams(NCAA)
    assert teams.by_espn_id("900001") == teams.by_espn_id("900002")


@pytest.fixture
def answers(monkeypatch):
    """Queue up answers for the prompts, in order."""

    def _queue(*replies: str) -> None:
        remaining = list(replies)
        monkeypatch.setattr("builtins.input", lambda _: remaining.pop(0))

    return _queue


def _candidate() -> NameCandidate:
    return NameCandidate(
        name="Army Cadets",
        known="Army Black Knights",
        similarity=0.95,
        gap_years=0,
        opponent_jaccard=0.8,
        name_years=(2004, 2005),
        known_years=(2006, 2025),
        name_games=100,
        known_games=60,
    )


def _activity() -> dict[str, TeamActivity]:
    return {
        "Army Cadets": TeamActivity("Army Cadets", {2004: {"Rival"}, 2005: {"Rival"}})
    }


def _namer() -> TeamNamer:
    return TeamNamer(default_teams(NCAA), "ncaafb")


def test_accepting_a_name_yields_one_observation_per_season(tmp_path, answers) -> None:
    answers("y")
    decisions = Decisions(tmp_path / "d.json", interactive=True)

    accepted = _review_names([_candidate()], _activity(), _namer(), decisions, "ncaafb")

    # One observation per season the name was actually seen, which is what
    # `name_in` reads to answer what a team was called back then.
    assert accepted == [
        ("349", _name("Army Cadets", 2004)),
        ("349", _name("Army Cadets", 2005)),
    ]


def test_rejecting_a_name_is_remembered(tmp_path, answers) -> None:
    """A second run shouldn't ask about the same wrong guess."""
    path = tmp_path / "d.json"
    answers("n")
    decisions = Decisions(path, interactive=True)
    assert (
        _review_names([_candidate()], _activity(), _namer(), decisions, "ncaafb") == []
    )
    decisions.save()

    # No answers queued: asking again would raise IndexError off the empty queue.
    reloaded = Decisions(path, interactive=True)
    assert (
        _review_names([_candidate()], _activity(), _namer(), reloaded, "ncaafb") == []
    )


def test_quitting_keeps_what_was_already_answered(tmp_path, answers) -> None:
    answers("y", "q")
    decisions = Decisions(tmp_path / "d.json", interactive=True)
    second = replace(_candidate(), name="Army Mules")
    activity = {
        **_activity(),
        "Army Mules": TeamActivity("Army Mules", {2000: set()}),
    }

    accepted = _review_names(
        [_candidate(), second], activity, _namer(), decisions, "ncaafb"
    )

    assert {o[1].name for o in accepted} == {"Army Cadets"}


def test_a_non_interactive_run_accepts_nothing(tmp_path) -> None:
    """There's no answer that accepts every guess unseen."""
    decisions = Decisions(tmp_path / "d.json", interactive=False)
    assert (
        _review_names([_candidate()], _activity(), _namer(), decisions, "ncaafb") == []
    )


def test_duplicate_ids_are_found_and_merged(tmp_path, answers) -> None:
    _record_all(
        [
            ("900001", _name("Bay State Otters", 2024)),
            ("900002", _name("Bay State Otters", 2025)),
        ],
        NCAA,
    )
    assert len(_duplicate_ids(default_teams(NCAA), "ncaafb")) == 1

    answers("y")
    decisions = Decisions(tmp_path / "d.json", interactive=True)
    merged = _review_duplicate_ids(default_teams(NCAA), "ncaafb", NCAA, decisions)

    assert merged == 1
    default_teams.cache_clear()
    # The older record is the duplicate: the newer id is the one ESPN is
    # still filing games under.
    assert default_teams(NCAA).by_espn_id("900001").espn_id == "900002"


def test_two_schools_that_share_a_name_stay_separate(tmp_path, answers) -> None:
    _record_all(
        [
            ("900001", _name("Coastal Bulldogs", 2025)),
            ("900002", _name("Coastal Bulldogs", 2025)),
        ],
        NCAA,
    )

    answers("n")
    decisions = Decisions(tmp_path / "d.json", interactive=True)
    assert _review_duplicate_ids(default_teams(NCAA), "ncaafb", NCAA, decisions) == 0

    default_teams.cache_clear()
    assert default_teams(NCAA).by_espn_id("900001") != default_teams(NCAA).by_espn_id(
        "900002"
    )
    assert decisions.distinct_ids

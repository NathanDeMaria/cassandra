from datetime import datetime

from call_it_what_you_want import teams_from_csv
from endgame.types import Game, Season, Week

from .alias_candidates import (
    check_registry,
    name_similarity,
    next_renames,
    normalize_name,
    propose_names,
    rows_for,
    team_activity,
    unresolved_names,
)
from .aliases import TeamNamer


def _season(year: int, *matchups: tuple[str, str]) -> Season:
    games = [
        Game(
            home=home,
            away=away,
            home_score=1,
            away_score=0,
            neutral_site=False,
            completed=True,
            date=datetime(year, 9, 1 + i),
            game_id=f"{year}-{i}",
        )
        for i, (home, away) in enumerate(matchups)
    ]
    return Season([Week(games, 1)], year)


def _namer(*rows: str) -> TeamNamer:
    return TeamNamer(teams_from_csv(["espn_id,name,year,source", *rows]))


_RENAME = [
    _season(2020, ("Old State Tigers", "Rival"), ("Old State Tigers", "Other")),
    _season(2021, ("Old State University Tigers", "Rival")),
]


def test_normalize_keeps_state() -> None:
    """`Michigan` and `Michigan State` are different schools."""
    assert normalize_name("Michigan Wolverines") != normalize_name(
        "Michigan State Spartans"
    )


def test_normalize_collapses_spelling_of_the_same_school() -> None:
    assert normalize_name("UW-Oshkosh Titans") == normalize_name(
        "Wisconsin-Oshkosh Titans"
    )
    assert (
        name_similarity("North Central Cardinals", "North Central College Cardinals")
        == 1.0
    )


def test_team_activity_tracks_years_and_opponents() -> None:
    activity = team_activity([_season(2020, ("A", "B")), _season(2021, ("A", "C"))])
    assert activity["A"].years == {2020, 2021}
    assert activity["A"].opponents_by_year[2021] == {"C"}
    assert activity["B"].first_year == activity["B"].last_year == 2020


def test_unresolved_names_are_what_the_registry_is_missing() -> None:
    namer = _namer("1,Old State University Tigers,2021,espn")
    unresolved = unresolved_names(team_activity(_RENAME), namer)
    assert "Old State University Tigers" not in unresolved
    assert "Old State Tigers" in unresolved


def test_proposes_a_rename() -> None:
    (candidate,) = propose_names(team_activity(_RENAME))
    # The name still in use is the known one: it's the one the registry is
    # most likely to already have an id for.
    assert candidate.name == "Old State Tigers"
    assert candidate.known == "Old State University Tigers"
    assert candidate.gap_years == 0
    assert candidate.opponent_jaccard > 0


def test_rows_attach_the_new_name_to_the_known_team() -> None:
    """One row per season the name was seen, which is what `name_in` reads."""
    namer = _namer("1,Old State University Tigers,2021,espn")
    activity = team_activity(_RENAME)
    (candidate,) = propose_names(activity, namer=namer)
    assert rows_for(candidate, activity, namer) == [
        ("1", "Old State Tigers", 2020, "espn")
    ]


def test_no_rows_when_neither_side_has_an_id() -> None:
    """A group the registry has never seen needs a team, not a name."""
    activity = team_activity(_RENAME)
    (candidate,) = propose_names(activity)
    assert rows_for(candidate, activity, TeamNamer.empty()) == []


def test_two_schools_playing_in_the_same_season_are_never_proposed() -> None:
    """The signal that separates a rename from a coincidence.

    These two names are similar enough that nothing about the strings tells
    them apart -- they're different schools, and the only evidence of it is
    that both were active at once.
    """
    seasons = [
        _season(2020, ("St. Thomas (MN) Tommies", "Rival")),
        _season(2020, ("St. Thomas (FL) Tommies", "Rival")),
        _season(2021, ("St. Thomas (MN) Tommies", "Rival")),
    ]
    assert propose_names(team_activity(seasons)) == []


def test_pairs_the_registry_already_agrees_about_are_skipped() -> None:
    namer = _namer(
        "1,Old State Tigers,2020,espn",
        "1,Old State University Tigers,2021,espn",
    )
    assert propose_names(team_activity(_RENAME), namer=namer) == []


def test_a_long_gap_is_not_a_rename() -> None:
    seasons = [
        _season(2000, ("Old State Tigers", "Rival")),
        _season(2021, ("Old State University Tigers", "Rival")),
    ]
    assert propose_names(team_activity(seasons), max_gap_years=2) == []


def test_check_registry_flags_two_programs_on_one_id() -> None:
    """The failure a registry can cause that no metric would show.

    Putting two schools that played each other on one id destroys both
    ratings, and the replay would run happily.
    """
    activity = team_activity(
        [_season(2020, ("St. Thomas (MN) Tommies", "St. Thomas (FL) Tommies"))]
    )
    namer = _namer(
        "1,St. Thomas (MN) Tommies,2020,espn",
        "1,St. Thomas (FL) Tommies,2020,espn",
    )
    (problem,) = check_registry(namer, activity)
    assert "same season" in problem


def test_check_registry_is_quiet_on_a_good_registry() -> None:
    namer = _namer(
        "1,Old State Tigers,2020,espn",
        "1,Old State University Tigers,2021,espn",
        "2,Rival,2020,espn",
        "3,Other,2020,espn",
    )
    assert check_registry(namer, team_activity(_RENAME)) == []


def test_next_renames_catches_a_debut_the_registry_lacks() -> None:
    """What keeps the registry from going stale after a data refresh."""
    namer = _namer("1,Old State Tigers,2020,espn")
    activity = team_activity(_RENAME)
    (suspect,) = next_renames(activity, namer, year=2021)
    assert suspect.name == "Old State Tigers"
    assert suspect.known == "Old State University Tigers"

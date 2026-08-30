from datetime import datetime, timezone

import pytest
from call_it_what_you_want import TeamNamer
from endgame.types import Game, Season, Week

from division_anchors import (
    TierGame,
    _first_years,
    _merge_thin_tiers,
    _tier_games,
    _Tiers,
    fit_tiers,
    main,
)


def _games(
    home: str, away: str, home_wins: int, away_wins: int, neutral: bool = True
) -> list[TierGame]:
    return [TierGame(home, away, 1.0, neutral)] * home_wins + [
        TierGame(home, away, 0.0, neutral)
    ] * away_wins


def test_a_tier_that_wins_more_rates_higher() -> None:
    fit = fit_tiers(_games("fbs", "fcs", 90, 10))

    assert fit.ratings["fbs"] > fit.ratings["fcs"]


def test_the_gap_matches_the_win_rate() -> None:
    """A 3:1 record is 10**(gap/400) = 3, so ~191 points."""
    fit = fit_tiers(_games("fbs", "fcs", 75, 25))

    assert fit.ratings["fbs"] - fit.ratings["fcs"] == pytest.approx(191, abs=5)


def test_even_tiers_land_together_at_the_mean() -> None:
    fit = fit_tiers(_games("fbs", "fcs", 50, 50), mean=1500)

    assert fit.ratings["fbs"] == pytest.approx(1500, abs=1)
    assert fit.ratings["fcs"] == pytest.approx(1500, abs=1)


def test_ratings_are_centred_on_the_mean() -> None:
    """Any constant added to every tier predicts the same, so one is chosen.

    Keeping the existing centre leaves teams where they are and only spreads
    the tiers around it.
    """
    fit = fit_tiers(_games("fbs", "fcs", 75, 25), mean=1500)

    assert sum(fit.ratings.values()) / 2 == pytest.approx(1500, abs=1)


def test_home_advantage_is_not_charged_to_the_smaller_division() -> None:
    """FBS buys home games against FCS, so the venue is confounded with tier.

    Two tiers that split evenly on a neutral floor, but where one hosts every
    non-neutral game and wins them, are equal -- the wins are the venue.
    """
    games = _games("fbs", "fcs", 50, 50) + _games("fbs", "fcs", 60, 40, neutral=False)

    fit = fit_tiers(games)

    assert fit.home_advantage > 0
    assert fit.ratings["fbs"] - fit.ratings["fcs"] == pytest.approx(0, abs=15)


def test_a_transitive_gap_is_recovered_without_direct_games() -> None:
    """D-III never plays FBS, so its scale has to come through FCS."""
    fit = fit_tiers(_games("fbs", "fcs", 75, 25) + _games("fcs", "d3", 75, 25))

    assert fit.ratings["fbs"] - fit.ratings["d3"] == pytest.approx(382, abs=15)


def test_no_games_is_not_a_crash() -> None:
    assert fit_tiers([]) == ({}, 0.0)


def test_a_thin_conference_falls_back_to_its_division() -> None:
    """A confident wrong anchor is worse than the shared one."""
    tiers = _Tiers(
        full={"Tiny": "fcs / Pioneer", "Big": "fbs / SEC"},
        division={"Tiny": "fcs", "Big": "fbs"},
    )

    resolved = _merge_thin_tiers(
        tiers,
        _games("fbs / SEC", "fcs / Pioneer", 3, 1) + _games("fbs / SEC", "x", 500, 0),
    )

    assert resolved == {"Tiny": "fcs", "Big": "fbs / SEC"}


def test_a_tier_with_no_games_at_all_falls_back() -> None:
    tiers = _Tiers(full={"Ghost": "d3 / Nowhere"}, division={"Ghost": "d3"})

    assert _merge_thin_tiers(tiers, []) == {"Ghost": "d3"}


def test_if_missing_leaves_an_existing_file_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The anchors job calls this on every run, so it has to be a no-op.

    Refitting each time would re-rate every model against a slightly different
    scale without anyone asking for it -- and it would spend the s3 read that
    `main` skips entirely to get here.
    """
    path = tmp_path / "ncaafb_division_anchors.json"
    path.write_text("{}")
    monkeypatch.setattr("division_anchors.anchor_path", lambda league: path)
    monkeypatch.setattr("division_anchors._build", _never_built)

    main(league="ncaafb", write=True, if_missing=True)


def test_if_missing_builds_when_there_is_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    built = []
    monkeypatch.setattr(
        "division_anchors.anchor_path", lambda league: tmp_path / "absent.json"
    )

    async def _record(league: str, write: bool) -> None:
        built.append((league, write))

    monkeypatch.setattr("division_anchors._build", _record)

    main(league="ncaafb", write=True, if_missing=True)

    assert built == [("ncaafb", True)]


async def _never_built(league: str, write: bool) -> None:
    raise AssertionError(f"rebuilt {league} anchors that already existed")


def _game(
    year: int,
    home: str = "Big",
    away: str = "Small",
    home_score: int = 10,
    away_score: int = 3,
    completed: bool = True,
) -> Game:
    return Game(
        home=home,
        home_score=home_score,
        away=away,
        away_score=away_score,
        neutral_site=False,
        completed=completed,
        date=datetime(year, 9, 2, tzinfo=timezone.utc),
        game_id=f"{home}-{away}-{year}",
    )


def _season(year: int, *games: Game) -> Season:
    return Season([Week(list(games), 1)], year)


def test_first_years_ignores_a_game_that_hasnt_been_played() -> None:
    """A fixture must not decide the tier a team is anchored against.

    A season pickle pulled in August carries September's schedule, so
    counting appearances would file every team under the coming season and
    judge its whole history against that year's classification.
    """
    seasons = [
        _season(2023, _game(2023, completed=False)),
        _season(2024, _game(2024)),
    ]

    assert _first_years(seasons, TeamNamer.empty()) == {"Big": 2024, "Small": 2024}


def test_tier_games_ignores_a_game_in_progress() -> None:
    """The gap between tiers is fit on final scores only.

    A game underway has a real, partial scoreline, so the tie guard below
    doesn't catch it -- whoever happens to be ahead at the moment of the
    fetch would otherwise be recorded as the winner.
    """
    tiers = {"Big": "fbs", "Small": "fcs"}
    seasons = [_season(2024, _game(2024, completed=False))]

    assert _tier_games(seasons, TeamNamer.empty(), tiers) == []

from datetime import datetime, timezone

from call_it_what_you_want import TeamNamer
from endgame.types import Game, Season, Week

from .exhibitions import (
    EXHIBITION_MAX_CAREER_GAMES,
    exhibition_count,
    exhibition_teams,
    without_exhibitions,
)


def _game(home: str, away: str, game_id: str, day: int, completed: bool = True) -> Game:
    return Game(
        home=home,
        home_score=49 if completed else 0,
        away=away,
        away_score=0,
        neutral_site=False,
        completed=completed,
        date=datetime(2023, 9, day, tzinfo=timezone.utc),
        game_id=game_id,
    )


def _season(games: list[Game], year: int = 2023) -> Season:
    return Season([Week(games, 1)], year)


def _schedule(team: str, n: int, start: int = 0) -> list[Game]:
    """`n` games for `team` against distinct, well-established opponents."""
    return [
        _game(team, f"Regular {start + i}", f"{team}-{start + i}", 1 + i % 28)
        for i in range(n)
    ]


def _regulars(n: int = 30) -> list[Game]:
    """Enough games among the regulars that none of them is a scrimmage team."""
    games = []
    for i in range(n):
        for j in range(EXHIBITION_MAX_CAREER_GAMES + 1):
            games.append(
                _game(f"Regular {i}", f"Regular {(i + j + 1) % n}", f"r{i}-{j}", 1 + j)
            )
    return games


def test_a_one_off_unanchored_opponent_is_an_exhibition() -> None:
    seasons = [_season([*_regulars(), _game("Regular 0", "Bible College", "x", 3)])]
    junk = exhibition_teams(seasons, anchored=[], namer=TeamNamer.empty())
    assert junk == {"Bible College"}


def test_an_anchored_team_is_a_program_however_few_its_games() -> None:
    seasons = [_season([*_regulars(), _game("Regular 0", "New D-III School", "x", 3)])]
    junk = exhibition_teams(
        seasons, anchored=["New D-III School"], namer=TeamNamer.empty()
    )
    assert junk == frozenset()


def test_an_unanchored_program_with_a_history_is_kept() -> None:
    """Point University, Wayland Baptist: NAIA, unfiled, and here every year."""
    games = _regulars() + _schedule("NAIA Program", EXHIBITION_MAX_CAREER_GAMES + 1)
    junk = exhibition_teams([_season(games)], anchored=[], namer=TeamNamer.empty())
    assert "NAIA Program" not in junk
    fewer = _regulars() + _schedule("Scrimmage Side", EXHIBITION_MAX_CAREER_GAMES)
    junk = exhibition_teams([_season(fewer)], anchored=[], namer=TeamNamer.empty())
    assert "Scrimmage Side" in junk


def test_career_games_span_seasons_and_count_fixtures() -> None:
    """A first-year program with a schedule is a program from its first Saturday."""
    played = _schedule("New Program", 2)
    fixtures = [
        _game("New Program", f"Regular {i}", f"fx{i}", 10 + i, completed=False)
        for i in range(4)
    ]
    this_year = _season([*_regulars(), *played, *fixtures])
    junk = exhibition_teams([this_year], anchored=[], namer=TeamNamer.empty())
    assert "New Program" not in junk
    # Two games a year for three years is six too, across seasons.
    years = [
        _season([*_regulars(), *_schedule("Perennial", 2, start=y)], 2020 + y)
        for y in range(3)
    ]
    junk = exhibition_teams(years, anchored=[], namer=TeamNamer.empty())
    assert "Perennial" not in junk


def test_without_exhibitions_drops_the_game_and_keeps_the_week() -> None:
    regulars = _regulars()
    season = _season([*regulars, _game("Regular 0", "Bible College", "x", 3)])
    cleaned = without_exhibitions([season], "test_league", namer=TeamNamer.empty())
    assert len(cleaned) == 1
    assert len(cleaned[0].weeks) == 1
    ids = {g.game_id for g in cleaned[0].weeks[0].games}
    assert "x" not in ids
    assert ids == {g.game_id for g in regulars}
    assert exhibition_count([season], cleaned) == 1
    assert cleaned[0].year == season.year


def test_a_league_with_no_exhibitions_comes_back_as_it_was() -> None:
    season = _season(_regulars())
    cleaned = without_exhibitions([season], "test_league", namer=TeamNamer.empty())
    assert cleaned == [season]
    assert exhibition_count([season], cleaned) == 0

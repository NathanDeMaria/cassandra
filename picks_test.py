from datetime import UTC, datetime, timedelta

from endgame.types import Game

from cassandra.qb_starts import AVAILABILITY_OUT, GameQb
from picks import QB_UNREAD_WINDOW, apply_rules, parse, qb_out_index, qb_reads

NOW = datetime(2026, 10, 21, 3, 0, tzinfo=UTC)  # Tue 22:00 CT


def _game(away: str, home: str, ago: timedelta, completed: bool = True) -> Game:
    return Game(
        home=home,
        home_score=0,
        away=away,
        away_score=0,
        neutral_site=False,
        completed=completed,
        date=NOW - ago,
        game_id=f"{away}@{home}",
    )


def test_only_recent_unread_games_are_read_off_espn() -> None:
    week = timedelta(days=7)
    games = [
        _game("Toledo", "Ohio", 2 * week),  # in the index: Toledo has plays
        _game("Ohio", "Kent", 2 * week),  # in the index: Kent has plays
        # A D-II opponent ESPN never had plays for, weeks ago: the sweeps
        # have run since, so it's missing for good and isn't read again.
        _game("Wheeling", "Toledo", week),
        _game("Akron", "Kent", timedelta(hours=3)),  # tonight's, not swept
        _game("Toledo", "Kent", -week, completed=False),
    ]
    stored = {"Toledo@Ohio", "Ohio@Kent"}
    upcoming = [games[-1]]
    unread, needy, wanted = qb_reads(games, stored, upcoming, NOW)
    assert unread == {"Akron@Kent"}
    assert needy == {"Kent"}
    # Kent's whole season, so the rule knows who its starter is.
    assert wanted == ["Akron@Kent", "Ohio@Kent"]
    # The old one is read while a sweep could still be behind on it.
    later = NOW - week + QB_UNREAD_WINDOW - timedelta(hours=1)
    unread, _, _ = qb_reads(games, stored, upcoming, later)
    assert "Wheeling@Toledo" in unread


def test_a_league_prices_with_its_own_model_unless_told() -> None:
    args = parse([])
    assert (args.league, args.model) == ("ncaafb", "glicko_margin_units_success")
    assert parse(["--model", "glicko_margin_units_roster"]).model == (
        "glicko_margin_units_roster"
    )

    womens = parse(["--league", "womens"])
    assert (womens.league, womens.model) == ("womens", "glicko_players")


def test_a_run_takes_its_leagues_rules_unless_told_otherwise() -> None:
    import pytest

    from cassandra.picks_job import LeagueRules

    rules = LeagueRules(
        kelly_fraction=0.3,
        max_bet=7.0,
        min_edge=0.06,
        max_since_entry_hours=20.0,
        max_overround=1.12,
        top=6,
        rungs=(1.0, 3.0),
    )
    args = parse(["--min-edge", "0.09"])
    apply_rules(args, rules)
    assert args.rule.min_edge == 0.09  # the flag over the rule
    assert args.rule.max_since_entry == timedelta(hours=20)
    assert (args.max_bet, args.top) == (7.0, 6)
    # The stake, its limits and the rungs only for a run that places.
    assert args.kelly_fraction is None and args.rungs == ()
    placing = parse(["--place"])
    apply_rules(placing, rules)
    assert placing.kelly_fraction == 0.3 and placing.rungs == (0.01, 0.03)
    assert apply_rules(parse(["--place", "--no-rungs"]), rules) is None
    with pytest.raises(SystemExit, match="no ncaafb rule for --min-edge"):
        apply_rules(parse([]), None)


def _qbs(starter: str) -> GameQb:
    return GameQb(
        starter=starter,
        attempts={starter: 30},
        names={starter: starter},
        snaps=[starter],
        last_snap={starter: (4, 3)},
    )


def test_a_live_index_prices_a_doubted_starter_unless_the_news_names_him() -> None:
    week = timedelta(days=7)
    played = [
        _game("Kent", "Ohio", n * week)._replace(game_id=f"g{n}") for n in (4, 3, 2)
    ]
    fixture = _game("Kent", "Ohio", -week, completed=False)._replace(game_id="next")
    starts = {g.game_id: {"Ohio": _qbs("ace"), "Kent": _qbs("kent")} for g in played}
    # Ohio's starter missed the last game
    starts[played[-1].game_id] = {"Ohio": _qbs("backup"), "Kent": _qbs("kent")}
    args = ("ncaafb", played + [fixture], [fixture], {})

    priced = qb_out_index(*args, {}, starts)
    named = qb_out_index(*args, {fixture.game_id: {"Ohio"}}, starts)

    assert not priced.is_out(fixture.game_id, "Ohio")
    assert 0 < priced.weight(fixture.game_id, "Ohio") < AVAILABILITY_OUT["missed"]
    assert priced.weight(fixture.game_id, "Kent") == 0.0
    assert named.is_out(fixture.game_id, "Ohio")
    assert named.weight(fixture.game_id, "Ohio") == 1.0

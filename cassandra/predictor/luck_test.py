from pathlib import Path

import pytest

from . import luck
from .adjustments import MatchupSources
from .conftest import GameFactory
from .glicko import GlickoPredictor
from .luck import GameLuck, LuckIndex, load_luck, validated_luck_share
from .qb_out import QbOutIndex
from .rest import RestLedger

INDEX = LuckIndex({"g1": GameLuck("Home U", "Away St", 6.0)})


def test_the_luck_is_toward_the_games_home_side(game: GameFactory) -> None:
    assert INDEX.home_luck(game("Home U", "Away St", game_id="g1")) == 6.0
    # filed the other way round, as a neutral site can be
    assert INDEX.home_luck(game("Away St", "Home U", game_id="g1")) == -6.0
    # one side named differently is still the game
    assert INDEX.home_luck(game("Home U", "Somebody", game_id="g1")) == 6.0
    assert INDEX.home_luck(game("Home U", "Away St", game_id="g2")) is None
    assert INDEX.home_luck(game("Other", "Teams", game_id="g1")) is None


def test_the_file_reads_by_canonical_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nfl_luck.csv"
    path.write_text(
        "league,season,game_id,date,home_team_id,home,away_team_id,away,luck,as_of\n"
        "nfl,2025,g1,2025-11-04,1,Home U,2,Away St,-3.5,2025-11-05\n"
    )
    monkeypatch.setattr(luck, "luck_path", lambda league: path)
    load_luck.cache_clear()
    try:
        assert load_luck("nfl") == {"g1": GameLuck("Home U", "Away St", -3.5)}
    finally:
        load_luck.cache_clear()


def test_a_share_outside_zero_to_one_is_refused() -> None:
    assert validated_luck_share(0.4) == 0.4
    with pytest.raises(ValueError):
        validated_luck_share(1.5)


def _predictor(share: float) -> GlickoPredictor:
    sources = MatchupSources(QbOutIndex(), RestLedger(), luck=INDEX)
    return GlickoPredictor(
        "test_league", scoring_method="sigmoid", luck_share=share, sources=sources
    )


def test_glicko_learns_less_from_a_lucky_win(game: GameFactory) -> None:
    won = game("Home U", "Away St", 80, 74, game_id="g1")
    plain, discounted = _predictor(0.0), _predictor(0.5)
    plain.update_game(won)
    discounted.update_game(won)
    # 6 points of it luck: half of that off, a 3-point win
    assert discounted._actual(won) == pytest.approx(
        plain._actual(game("Home U", "Away St", 77, 74, game_id="g1"))
    )
    assert discounted.get_rating("Home U").rating < plain.get_rating("Home U").rating
    # and predicts the game the same: the luck is in the update alone
    fresh = game("Home U", "Away St", game_id="g1")
    assert _predictor(0.5).predict_game(fresh) == _predictor(0.0).predict_game(fresh)


def test_the_share_is_kept_only_when_on() -> None:
    restored = GlickoPredictor.from_state_dict(_predictor(0.4).state_dict())
    assert restored.state_dict()["luck_share"] == 0.4
    assert "luck_share" not in _predictor(0.0).state_dict()


def test_a_share_needs_a_scoring_method_that_reads_the_margin() -> None:
    with pytest.raises(ValueError, match="sigmoid"):
        GlickoPredictor("test_league", scoring_method="binary", luck_share=0.4)


FLOWED = LuckIndex({"g1": GameLuck("Home U", "Away St", 6.0, 4.0, -8.0, 2.0)})


def test_the_flow_readings_turn_with_the_luck(game: GameFactory) -> None:
    found = FLOWED.toward_home(game("Away St", "Home U", game_id="g1"))
    assert found == GameLuck("Away St", "Home U", -6.0, -4.0, 8.0, -2.0)
    # a game filed without them turns to nothing
    assert INDEX.toward_home(game("Away St", "Home U", game_id="g1")) == GameLuck(
        "Away St", "Home U", -6.0
    )


def test_the_file_reads_its_flow_columns_where_it_has_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nfl_luck.csv"
    path.write_text(
        "league,season,game_id,date,home_team_id,home,away_team_id,away,luck,as_of,"
        "jumpers,over_lead,closing,source\n"
        "nfl,2025,g1,2025-11-04,1,Home U,2,Away St,-3.5,2025-11-05,1.5,-4,2,shots\n"
        "nfl,2010,g2,2010-11-04,1,Home U,2,Away St,2.5,2010-11-05,,,,box\n"
    )
    monkeypatch.setattr(luck, "luck_path", lambda league: path)
    load_luck.cache_clear()
    try:
        assert load_luck("nfl") == {
            "g1": GameLuck("Home U", "Away St", -3.5, 1.5, -4.0, 2.0),
            "g2": GameLuck("Home U", "Away St", 2.5),
        }
    finally:
        load_luck.cache_clear()


def _flowed(
    luck_share: float = 0.0,
    jumper_share: float = 0.0,
    lead_share: float = 0.0,
    closing_share: float = 0.0,
) -> GlickoPredictor:
    sources = MatchupSources(QbOutIndex(), RestLedger(), luck=FLOWED)
    return GlickoPredictor(
        "test_league",
        scoring_method="sigmoid",
        sources=sources,
        luck_share=luck_share,
        jumper_share=jumper_share,
        lead_share=lead_share,
        closing_share=closing_share,
    )


@pytest.mark.parametrize(
    "shares, counted_as",
    [
        ({"jumper_share": 0.5}, 78),  # 4 points of jumpers, half off
        ({"lead_share": 0.25}, 82),  # the margin 8 short of twice the lead
        ({"closing_share": 0.5}, 79),  # 2 points in the last five minutes
        ({"luck_share": 0.5, "jumper_share": 0.5, "closing_share": 0.5}, 74),
    ],
)
def test_each_share_takes_its_own_reading_off_the_margin(
    game: GameFactory, shares: dict[str, float], counted_as: int
) -> None:
    won = game("Home U", "Away St", 80, 74, game_id="g1")
    plain = _flowed()
    assert _flowed(**shares)._actual(won) == pytest.approx(
        plain._actual(game("Home U", "Away St", counted_as, 74, game_id="g2"))
    )


def test_a_game_without_a_reading_keeps_that_much_of_its_margin(
    game: GameFactory,
) -> None:
    won = game("Home U", "Away St", 80, 74, game_id="g1")
    sources = MatchupSources(QbOutIndex(), RestLedger(), luck=INDEX)
    unread = GlickoPredictor(
        "test_league", scoring_method="sigmoid", jumper_share=0.5, sources=sources
    )
    assert unread._actual(won) == pytest.approx(_predictor(0.0)._actual(won))


def test_flow_shares_are_kept_only_when_on_and_need_a_margin() -> None:
    state = _flowed(jumper_share=0.4, lead_share=0.25).state_dict()
    assert (state["jumper_share"], state["lead_share"]) == (0.4, 0.25)
    assert "closing_share" not in state and "luck_share" not in state
    restored = GlickoPredictor.from_state_dict(state)
    assert restored.state_dict()["jumper_share"] == 0.4
    with pytest.raises(ValueError, match="sigmoid"):
        GlickoPredictor("test_league", scoring_method="binary", lead_share=0.2)
    with pytest.raises(ValueError, match="jumper_share"):
        GlickoPredictor("test_league", scoring_method="sigmoid", jumper_share=1.5)

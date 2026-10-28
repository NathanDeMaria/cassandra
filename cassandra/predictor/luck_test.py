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

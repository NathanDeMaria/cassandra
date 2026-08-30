"""What every predictor has to do, run once against each of them.

These were copy-pasted into each model's test file, which meant a new model
started out with whichever subset of them its author remembered, and a change
to a shared seam -- `regress`, `anchor`, `state_dict` -- had to be re-argued
in three places. How a particular model turns a game into a rating stays in
that model's own test file; what's here is what a caller gets from any of
them.

Adding a model means adding it to MODELS (and to RATED_MODELS if it keeps
per-team ratings), not writing these again.
"""

import json
from pathlib import Path

import pytest

from .base_predictor import MEAN_RATING
from .conftest import GameFactory
from .elo import EloPredictor
from .elo538 import Elo538Predictor
from .flat import FlatPredictor
from .glicko import GlickoPredictor, _Rating

# Spelled as a union of concrete classes rather than `type[Predictor]` so the
# keyword arguments the contract passes -- `anchors`, `season_regression` --
# are checked against the constructors that actually declare them. See
# `validated_regression` for why they aren't in the base signature.
RatedModel = type[EloPredictor] | type[Elo538Predictor] | type[GlickoPredictor]
Rated = EloPredictor | Elo538Predictor | GlickoPredictor
Model = RatedModel | type[FlatPredictor]

RATED_MODELS: list[RatedModel] = [EloPredictor, Elo538Predictor, GlickoPredictor]
MODELS: list[Model] = [*RATED_MODELS, FlatPredictor]

any_model = pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
rated_model = pytest.mark.parametrize("model", RATED_MODELS, ids=lambda m: m.__name__)


def _rating(predictor: Rated, team: str) -> float:
    """One team's rating as a plain number.

    Elo keeps a float and Glicko a (rating, deviation) pair; everything here
    asks about the number both of them have. The deviation is Glicko's own
    business and glicko_test.py is where it's checked.
    """
    rating = predictor.get_rating(team)
    return rating.rating if isinstance(rating, _Rating) else rating


@any_model
def test_save_load_round_trips_the_predictions(
    model: Model, tmp_path: Path, game: GameFactory
) -> None:
    """Not just the ratings: what a release is for is the next prediction."""
    predictor = model("test_league")
    predictor.update_game(game("Team A", "Team B", 2, 1))
    predictor.update_game(game("Team C", "Team A", 0, 3))

    path = tmp_path / "state.json"
    predictor.save_state(path)
    loaded = model.load_state(path)

    upcoming = game("Team A", "Team C")
    assert loaded.predict_game(upcoming).team1_win_prob == pytest.approx(
        predictor.predict_game(upcoming).team1_win_prob
    )


@any_model
def test_the_state_dict_is_plain_json(model: Model, game: GameFactory) -> None:
    """save_state/load_state are one serialization of this, not a second format.

    A caller that already holds the data -- a web service reading a release --
    goes through state_dict/from_state_dict instead of writing a temp file to
    read it back, so that path has to work on its own.
    """
    predictor = model("test_league")
    predictor.update_game(game("Team A", "Team B", 1, 0))

    state = predictor.state_dict()
    assert json.loads(json.dumps(state)) == state

    restored = model.from_state_dict(state)
    upcoming = game("Team A", "Team B")
    assert restored.predict_game(upcoming).team1_win_prob == pytest.approx(
        predictor.predict_game(upcoming).team1_win_prob
    )


@rated_model
def test_a_win_moves_the_winner_up_and_the_loser_down(
    model: RatedModel, game: GameFactory
) -> None:
    predictor = model("test_league")
    predictor.update_game(game("Team A", "Team B", 1, 0))

    assert _rating(predictor, "Team A") > MEAN_RATING > _rating(predictor, "Team B")


@rated_model
def test_an_unknown_team_sits_at_the_mean(model: RatedModel) -> None:
    assert _rating(model("test_league"), "Nobody Has Heard Of Them") == MEAN_RATING


@rated_model
def test_an_unplayed_team_starts_at_its_anchor(model: RatedModel) -> None:
    """The one moment a division gap can enter a rating.

    Once a D-III team has played, every result it has is against other D-III
    teams, so nothing downstream can put it on the same scale as an FBS team.
    """
    predictor = model("test_league", anchors={"Team A": 1200})

    assert _rating(predictor, "Team A") == 1200
    assert _rating(predictor, "Team B") == MEAN_RATING


@rated_model
def test_no_regression_by_default(model: RatedModel, game: GameFactory) -> None:
    """The default has to be a no-op: every release published so far omits it."""
    predictor = model("test_league")
    predictor.update_game(game("Team A", "Team B", 1, 0))
    before = _rating(predictor, "Team A")

    predictor.pass_season()

    assert _rating(predictor, "Team A") == before


@rated_model
def test_season_regression_pulls_both_directions_toward_the_mean(
    model: RatedModel, game: GameFactory
) -> None:
    predictor = model("test_league", season_regression=0.5)
    predictor.update_game(game("Team A", "Team B", 1, 0))
    winner = _rating(predictor, "Team A")
    loser = _rating(predictor, "Team B")

    predictor.pass_season()

    assert _rating(predictor, "Team A") == pytest.approx(
        MEAN_RATING + (winner - MEAN_RATING) / 2
    )
    assert _rating(predictor, "Team B") == pytest.approx(
        MEAN_RATING + (loser - MEAN_RATING) / 2
    )


@rated_model
def test_full_regression_lands_on_the_anchor(
    model: RatedModel, game: GameFactory
) -> None:
    """The seam the per-division priors come in through.

    A team whose anchor is 1200 forgets the season back to 1200, not to the
    1500 that only makes sense for a league whose teams all play each other.
    """
    predictor = model("test_league", season_regression=1.0, anchors={"Team A": 1200})
    predictor.update_game(game("Team A", "Team B", 1, 0))

    predictor.pass_season()

    assert _rating(predictor, "Team A") == pytest.approx(1200)
    assert _rating(predictor, "Team B") == pytest.approx(MEAN_RATING)


@rated_model
def test_the_anchor_still_bites_with_regression_switched_off(
    model: RatedModel, game: GameFactory
) -> None:
    """The case that matters: `season_regression` tunes to 0 in every ncaafb model.

    `regress` is the only other thing that reads an anchor, and at 0 it's a
    no-op -- so if the starting rating didn't come from the anchor, the whole
    division fit would be inert exactly where it was built to be used.
    """
    predictor = model("test_league", season_regression=0.0, anchors={"Team A": 1200})

    predictor.update_game(game("Team A", "Team B", 1, 0))
    predictor.pass_season()

    assert _rating(predictor, "Team A") < MEAN_RATING


@rated_model
def test_out_of_range_regression_is_rejected(model: RatedModel) -> None:
    with pytest.raises(ValueError):
        model("test_league", season_regression=1.5)
    with pytest.raises(ValueError):
        model("test_league", season_regression=-0.1)


@rated_model
def test_season_regression_round_trips(
    model: RatedModel, tmp_path: Path, game: GameFactory
) -> None:
    predictor = model("test_league", season_regression=0.25)
    predictor.update_game(game("Team A", "Team B", 1, 0))
    path = tmp_path / "state.json"
    predictor.save_state(path)

    loaded = model.load_state(path)
    loaded.pass_season()
    predictor.pass_season()

    assert _rating(loaded, "Team A") == pytest.approx(_rating(predictor, "Team A"))


@rated_model
def test_anchors_round_trip_through_the_state_dict(model: RatedModel) -> None:
    """A release replays against the anchors it was fit with.

    Recomputing them on load instead would silently re-rate a published model
    whenever the anchor file changed underneath it.
    """
    predictor = model("test_league", anchors={"Team A": 1200})

    restored = model.from_state_dict(predictor.state_dict())

    assert restored._anchors == {"Team A": 1200}


@rated_model
def test_a_promoted_team_is_anchored_where_it_moved_to(model: RatedModel) -> None:
    """The reason an anchor can carry a history at all.

    North Dakota State was D-II in 2002 and has been the best team in FCS
    ever since. Anchored at the division it started in, it enters the replay
    hundreds of points below the teams it plays and is pulled back down there
    every offseason after the move.
    """
    predictor = model("test_league", anchors={"Team A": [[2002, 1200], [2004, 1500]]})

    assert _rating(predictor, "Team A") == 1200

    predictor.pass_season(2004)

    assert _rating(predictor, "Team A") == 1500


@rated_model
def test_an_anchor_history_is_read_at_the_season_in_hand(model: RatedModel) -> None:
    """Each step holds until the next one starts, and the ends clamp."""
    predictor = model("test_league", anchors={"Team A": [[2002, 1200], [2010, 1500]]})

    for year, expected in ((1998, 1200), (2002, 1200), (2009, 1200), (2010, 1500)):
        predictor.pass_season(year)
        assert _rating(predictor, "Team A") == expected, year

    # Past the last step -- an offseason rollover into a season nobody has
    # played -- holds the last one rather than falling off the end.
    predictor.pass_season(2030)
    assert _rating(predictor, "Team A") == 1500


@rated_model
def test_rolling_over_without_a_year_keeps_the_season(model: RatedModel) -> None:
    """`publish` rolls into a season it can't name; that must not reset the
    clock to a promoted team's first division."""
    predictor = model("test_league", anchors={"Team A": [[2002, 1200], [2004, 1500]]})
    predictor.pass_season(2004)

    predictor.pass_season()

    assert _rating(predictor, "Team A") == 1500


@rated_model
def test_the_clock_moves_before_the_regression(
    model: RatedModel, game: GameFactory
) -> None:
    """A team regresses toward where it is about to play, not where it was.

    Full regression makes the ordering visible: after crossing into 2004 the
    rating has to land on the 2004 anchor, not the 2002 one.
    """
    predictor = model(
        "test_league",
        season_regression=1.0,
        anchors={"Team A": [[2002, 1200], [2004, 1500]]},
    )
    predictor.update_game(game("Team A", "Team B", 1, 0))

    predictor.pass_season(2004)

    assert _rating(predictor, "Team A") == pytest.approx(1500)


@rated_model
def test_an_anchor_history_round_trips_through_the_state_dict(
    model: RatedModel,
) -> None:
    steps = [[2002, 1200], [2004, 1500]]
    predictor = model("test_league", anchors={"Team A": steps})

    restored = model.from_state_dict(json.loads(json.dumps(predictor.state_dict())))

    assert restored._anchors == {"Team A": steps}
    assert _rating(restored, "Team A") == 1200


@rated_model
def test_explicit_empty_anchors_are_not_replaced_by_the_saved_ones(
    model: RatedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`{}` means no anchors, not "go find some"."""
    monkeypatch.setattr(
        "cassandra.predictor.base_predictor.load_anchors",
        lambda league: {"Team A": 1200},
    )

    assert model("test_league", anchors={})._anchors == {}
    assert model("test_league")._anchors == {"Team A": 1200}

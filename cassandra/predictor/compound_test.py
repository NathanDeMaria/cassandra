"""Glicko with an offense and a defense per team.

The contract -- save, load, regress, anchor -- is `contract_test`'s. What's
here is the compound part: that the children are rated on EPA as two Glicko
contests in the parent's own currency, that they sit on the anchor (or the
parent) as a prior, that they speak at prediction through one blend weight
faded by how much they know, and that with the weight off or the index empty
the parent is all there is.
"""

import json
from collections.abc import Mapping

import pytest

from .base_predictor import MEAN_RATING, Anchor
from .compound import (
    DEFAULT_DEFENSE_SCALE,
    DEFAULT_OFFENSE_SCALE,
    DEFAULT_PARENT_SHARE,
    DEFAULT_UNIT_WEIGHT,
    CompoundGlickoPredictor,
)
from .conftest import GameFactory
from .epa import EpaIndex
from .glicko import GlickoPredictor
from .opponent_prior import OpponentPriorManager
from .types import GameEpa, Rating, Unit


def _epa(home: float, away: float) -> GameEpa:
    """A game's EPA where the two readings agree and the samples are full.

    The compound model reads only the weighted pair; the flat pair is here
    because the tuple requires it. Nothing in these tests is about the
    counts or the weights.
    """
    return GameEpa(
        home=home,
        away=away,
        home_plays=70,
        away_plays=70,
        home_weighted=home,
        away_weighted=away,
        home_weight=60.0,
        away_weight=60.0,
    )


# A game where the home offense moved the ball and the away one didn't.
LOPSIDED = _epa(0.3, -0.3)


def _predictor(
    epa: dict[str, GameEpa] | None = None,
    *,
    unit_weight: float = DEFAULT_UNIT_WEIGHT,
    offense_scale: float = DEFAULT_OFFENSE_SCALE,
    defense_scale: float = DEFAULT_DEFENSE_SCALE,
    parent_share: float = DEFAULT_PARENT_SHARE,
    offense_initial_rd: float | None = None,
    defense_initial_rd: float | None = None,
    offense_weekly_rd_increase: float | None = None,
    defense_weekly_rd_increase: float | None = None,
    offense_season_rd_increase: float | None = None,
    defense_season_rd_increase: float | None = None,
    initial_rd: float = 216,
    home_advantage: float = 95,
    weekly_rd_increase: float = 1,
    season_regression: float = 0.0,
    anchors: Mapping[str, Anchor] | None = None,
) -> CompoundGlickoPredictor:
    return CompoundGlickoPredictor(
        "test_league",
        game_epa=EpaIndex(epa or {}),
        unit_weight=unit_weight,
        offense_scale=offense_scale,
        defense_scale=defense_scale,
        parent_share=parent_share,
        offense_initial_rd=offense_initial_rd,
        defense_initial_rd=defense_initial_rd,
        offense_weekly_rd_increase=offense_weekly_rd_increase,
        defense_weekly_rd_increase=defense_weekly_rd_increase,
        offense_season_rd_increase=offense_season_rd_increase,
        defense_season_rd_increase=defense_season_rd_increase,
        initial_rd=initial_rd,
        home_advantage=home_advantage,
        weekly_rd_increase=weekly_rd_increase,
        season_regression=season_regression,
        anchors=anchors,
    )


def test_unit_weight_zero_is_glicko_game_by_game(game: GameFactory) -> None:
    """The baseline a search's own zero has to reproduce, on the same games.

    Every game has EPA, so the children *are* moving -- they just aren't
    allowed to speak. That is the stronger version of the check: it says the
    children's update touches nothing the parent reads.
    """
    epa = {str(i): LOPSIDED for i in range(4)}
    compound = _predictor(epa, unit_weight=0.0)
    glicko = GlickoPredictor("test_league")
    schedule = [
        game("A", "B", 21, 7, game_id="0"),
        game("C", "D", 10, 24, game_id="1"),
        game("A", "C", 14, 14, game_id="2"),
        game("D", "B", 3, 30, game_id="3"),
    ]
    for played in schedule:
        assert compound.update_game(played) == glicko.update_game(played)

    assert compound.unit_information("A") > 0
    assert _parents(compound) == glicko.ratings
    assert compound.predict_game(game("B", "C")) == glicko.predict_game(game("B", "C"))


def _parents(predictor: CompoundGlickoPredictor) -> dict[str, Rating]:
    """The ratings without the sides, for comparing against a plain Glicko."""
    return {
        team: rating._replace(units=None) for team, rating in predictor.ratings.items()
    }


def test_a_league_without_epa_is_glicko_whatever_the_weight(
    game: GameFactory,
) -> None:
    """Four of the six leagues, and every ncaafb season before 2006.

    Not because the weight is off: because a unit nobody has seen has
    earned no precision, so the combination has nothing to combine.
    """
    compound = _predictor(unit_weight=5.0)
    glicko = GlickoPredictor("test_league")
    for played in (game("A", "B", 21, 7), game("B", "C", 3, 10)):
        assert compound.update_game(played) == glicko.update_game(played)

    assert compound.unit_information("A") == 0
    assert compound.predict_game(game("C", "A")) == glicko.predict_game(game("C", "A"))


def test_the_offense_that_moved_the_ball_goes_up_and_the_defense_it_faced_goes_down(
    game: GameFactory,
) -> None:
    predictor = _predictor({"g": LOPSIDED})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    home, away = predictor.get_units("A"), predictor.get_units("B")
    assert home.offense.rating > MEAN_RATING > away.defense.rating
    # The other contest went the other way: B's offense averaged -0.3.
    assert away.offense.rating < MEAN_RATING < home.defense.rating


def test_a_contest_tightens_both_units_deviations_and_raises_confidence(
    game: GameFactory,
) -> None:
    predictor = _predictor({"g": LOPSIDED})
    assert predictor.unit_information("A") == 0

    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    for team in ("A", "B"):
        units = predictor.get_units(team)
        assert units.offense.rating_deviation < predictor._offense_initial_rd
        assert units.defense.rating_deviation < predictor._defense_initial_rd
        assert predictor.unit_information(team) > 0


def test_units_sit_on_the_anchor_before_they_have_played(game: GameFactory) -> None:
    """The reason the children are on the parent's scale at all.

    A division fit for a team is a prior for both of its sides, so a D-III
    offense enters where a D-III team does rather than at a 0 that means
    nothing next to an FBS defense.
    """
    predictor = _predictor(anchors={"A": 1200})

    units = predictor.get_units("A")
    assert units.offense.rating == units.defense.rating == 1200
    assert predictor.unit_rating("A") == 1200
    assert predictor.unit_rating("B") == MEAN_RATING


def test_at_full_parent_share_units_sit_on_the_parent(game: GameFactory) -> None:
    predictor = _predictor(parent_share=1.0)
    predictor.update_game(game("A", "B", 21, 7))  # no EPA: only the parent moves
    parent = predictor.get_rating("A").rating
    assert parent != MEAN_RATING

    assert predictor.unit_rating("A") == parent


def test_at_zero_parent_share_the_children_never_see_the_parent(
    game: GameFactory,
) -> None:
    """The default: what the record says about a team changes nothing about
    how its units read an EPA line. The children are then a second opinion
    rather than a correction to the first."""
    epa = {"g": LOPSIDED}
    fresh, seasoned = _predictor(epa), _predictor(epa)
    seasoned.update_game(game("A", "Z", 42, 0))  # A's parent moves; no EPA
    assert seasoned.get_rating("A") != fresh.get_rating("A")

    played = game("A", "B", 21, 7, game_id="g")
    fresh.update_game(played)
    seasoned.update_game(played)

    assert fresh._offsets("A") == seasoned._offsets("A")
    assert fresh._offsets("B") == seasoned._offsets("B")


def test_at_full_parent_share_the_children_are_residuals(game: GameFactory) -> None:
    """The same EPA line earns a strong team's offense less than a weak team's.

    A team rated 400 points above its opponent is *expected* to move the ball
    on it, so doing so is less of a surprise and its offense moves less past
    its prior. In the parent's own currency that expectation is reachable --
    +0.3 per play over 70 snaps is 21 points, and the parent reads 21 points
    as a decisive win -- so this is a smaller step in the same direction,
    not the reversal the arbitrary-currency version of this model produced.
    """
    epa = {"g": LOPSIDED}
    favored = _predictor(epa, parent_share=1.0, anchors={"A": 1700, "B": 1300})
    even = _predictor(epa, parent_share=1.0)
    played = game("A", "B", 21, 7, game_id="g")
    favored.update_game(played)
    even.update_game(played)

    assert 0 < favored._offsets("A").offense.rating < even._offsets("A").offense.rating


def test_the_children_are_scored_against_the_parent_before_the_game(
    game: GameFactory,
) -> None:
    """The expected score uses who the opponent *was*, not who the result made them.

    Visible by comparison, at a `parent_share` where the parent is read at
    all: hand the update a parent that has already moved and the offsets
    come out different.
    """
    epa = {"g": LOPSIDED}
    predictor = _predictor(epa, parent_share=1.0)
    played = game("A", "B", 21, 7, game_id="g")
    predictor.update_game(played)

    stale = _predictor(epa, parent_share=1.0)
    GlickoPredictor.update_game(stale, played)
    # Now feed the children the *post-game* parents, which is the ordering
    # `update_game` exists to avoid.
    stale._update_units(played, stale.get_rating("A"), stale.get_rating("B"))

    assert predictor._offsets("A") != stale._offsets("A")


def test_the_children_move_the_prediction(game: GameFactory) -> None:
    """Same records, different shapes, different lines.

    A and B each beat an unrated opponent by the same score, so the parent
    can't tell them apart. A did it by moving the ball; B did it without.
    The compound model favors A when they meet, and by more as the weight
    goes up.
    """
    epa = {
        "a": _epa(0.4, -0.4),
        "b": _epa(-0.1, 0.1),
    }
    predictions = []
    for weight in (0.0, 1.0, 4.0):
        predictor = _predictor(epa, unit_weight=weight)
        predictor.update_game(game("A", "C", 21, 7, game_id="a"))
        predictor.update_game(game("B", "D", 21, 7, game_id="b"))
        assert _parents(predictor)["A"] == _parents(predictor)["B"]
        predictions.append(predictor.predict_game(game("A", "B")).team1_win_prob)

    assert (
        predictions[0]
        == GlickoPredictor("test_league").predict_game(game("A", "B")).team1_win_prob
        == pytest.approx(0.633, abs=0.001)
    )
    assert predictions[0] < predictions[1] < predictions[2]


def test_a_team_the_index_has_never_seen_keeps_its_parent_rating(
    game: GameFactory,
) -> None:
    """The blend is per team, faded by that team's own confidence.

    A has EPA and B doesn't. A's line is blended; B's is its record exactly,
    not pulled toward an anchor by a unit that is nothing but the anchor.
    """
    predictor = _predictor({"g": LOPSIDED}, anchors={"B": 1200})
    predictor.update_game(game("A", "C", 21, 7, game_id="g"))
    predictor.update_game(game("B", "D", 21, 7))  # no EPA

    assert predictor._blended_rating("B") == predictor.get_rating("B").rating
    assert predictor._blended_rating("A") != predictor.get_rating("A").rating


def test_the_unit_gap_flips_with_the_venue(game: GameFactory) -> None:
    predictor = _predictor({"g": LOPSIDED})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    assert predictor.unit_gap(game("A", "B")) == pytest.approx(
        -predictor.unit_gap(game("B", "A"))
    )


def test_a_contest_is_a_logistic_of_epa_per_play_at_the_sides_own_scale() -> None:
    """+0.1 a play at 10 logits per point is one logit over an average offense
    (the center is 0 before any game). No snap count anywhere: the same
    per-play average scores the same whether the offense ran 40 plays or 90."""
    predictor = _predictor()
    assert predictor._contest_score(0.1, 10.0) == pytest.approx(
        1 / (1 + 2.718281828**-1)
    )
    assert predictor._contest_score(0.1, 20.0) == pytest.approx(
        1 / (1 + 2.718281828**-2)
    )


def test_each_side_reads_the_same_contest_at_its_own_sharpness(
    game: GameFactory,
) -> None:
    """A sharper defense scale moves defenses more, and offenses not at all."""
    played = game("A", "B", 21, 7, game_id="g")
    even = _predictor({"g": LOPSIDED}, offense_scale=10.0, defense_scale=10.0)
    sharp_defense = _predictor({"g": LOPSIDED}, offense_scale=10.0, defense_scale=30.0)
    even.update_game(played)
    sharp_defense.update_game(played)

    assert sharp_defense.get_units("A").offense == even.get_units("A").offense
    assert sharp_defense.get_units("A").defense.rating > (
        even.get_units("A").defense.rating
    )
    assert sharp_defense.get_units("B").defense.rating < (
        even.get_units("B").defense.rating
    )


def test_the_contest_reads_the_garbage_time_adjusted_average(
    game: GameFactory,
) -> None:
    """The weighted pair, not the flat one, and never the snap count.

    Two games whose flat readings and play counts differ every way they can
    and whose weighted readings agree move the units identically.
    """
    lopsided = GameEpa(
        home=0.3,
        away=-0.3,
        home_plays=70,
        away_plays=70,
        home_weighted=0.2,
        away_weighted=-0.1,
        home_weight=50.0,
        away_weight=45.0,
    )
    same_when_contested = GameEpa(
        home=0.0,
        away=0.1,
        home_plays=90,
        away_plays=40,
        home_weighted=0.2,
        away_weighted=-0.1,
        home_weight=20.0,
        away_weight=12.0,
    )
    first = _predictor({"g": lopsided})
    second = _predictor({"g": same_when_contested})
    first.update_game(game("A", "B", 21, 7, game_id="g"))
    second.update_game(game("A", "B", 21, 7, game_id="g"))

    assert first.get_units("A") == second.get_units("A")
    assert first.get_units("B") == second.get_units("B")


def test_a_game_with_no_weighted_reading_runs_neither_contest(
    game: GameFactory,
) -> None:
    """A side whose every snap came with the game decided has nothing to
    average, and half a game can't be rated as a whole one."""
    decided = GameEpa(home=0.3, away=-0.3, home_plays=70, away_plays=70)
    predictor = _predictor({"g": decided})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    assert predictor.unit_information("A") == 0
    assert predictor.epa_center == 0


def test_the_center_is_the_running_mean_of_every_offense_seen(
    game: GameFactory,
) -> None:
    predictor = _predictor(
        {
            "g": _epa(0.1, -0.3),
            "h": _epa(0.0, 0.0),
        }
    )
    assert predictor.epa_center == 0

    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    assert predictor.epa_center == pytest.approx(-0.1)

    predictor.update_game(game("C", "D", 21, 7, game_id="h"))
    assert predictor.epa_center == pytest.approx(-0.05)


def test_two_average_offenses_leave_the_units_where_they_were(
    game: GameFactory,
) -> None:
    """A game centered on itself, with no home edge, is two drawn contests."""
    epa = {"g": _epa(-0.05, -0.05)}
    predictor = _predictor(epa, home_advantage=0)
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    for team in ("A", "B"):
        assert predictor.get_units(team).offense.rating == pytest.approx(MEAN_RATING)
        assert predictor.get_units(team).defense.rating == pytest.approx(MEAN_RATING)


def test_the_children_round_trip_through_the_state_dict(game: GameFactory) -> None:
    """The path a release would have to take for the children to survive."""
    predictor = _predictor(
        {"g": LOPSIDED}, unit_weight=0.3, defense_scale=9.0, parent_share=0.4
    )
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    state = json.loads(json.dumps(predictor.state_dict()))
    restored = CompoundGlickoPredictor.from_state_dict(state)

    assert restored.get_units("A") == predictor.get_units("A")
    assert restored.get_units("B") == predictor.get_units("B")
    assert restored.epa_center == predictor.epa_center
    assert (restored.unit_weight, restored.parent_share) == (0.3, 0.4)
    assert restored.predict_game(game("B", "A")) == predictor.predict_game(
        game("B", "A")
    )


def test_ratings_carry_the_sides_as_absolutes_for_teams_that_have_them(
    game: GameFactory,
) -> None:
    """What a release reads: the two sides on the team's scale, or nothing."""
    predictor = _predictor({"g": LOPSIDED}, anchors={"A": 1200})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    predictor.update_game(game("C", "D", 21, 7))  # no EPA

    sides = predictor.ratings["A"].units
    assert sides is not None
    assert sides.offense == Unit(*predictor.get_units("A").offense)
    # Absolutes on the team's scale, not offsets: both of A's sides won
    # their contests, so both sit above the 1200 anchor rather than above 0.
    assert sides.offense.rating > 1200
    assert sides.defense.rating > 1200
    assert predictor.ratings["C"].units is None


def test_from_ratings_puts_the_sides_back(game: GameFactory) -> None:
    """The release round trip, side by side with the parent's."""
    predictor = _predictor({"g": LOPSIDED}, anchors={"A": 1200})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    predictor.update_game(game("C", "D", 21, 7))  # no EPA

    rebuilt = CompoundGlickoPredictor.from_ratings(
        "test_league", predictor.ratings, game_epa=EpaIndex(), anchors={"A": 1200}
    )

    for team in ("A", "B"):
        assert rebuilt.get_units(team) == predictor.get_units(team)
        assert rebuilt.unit_information(team) == predictor.unit_information(team)
    assert rebuilt.unit_information("C") == 0
    assert rebuilt.ratings == predictor.ratings
    for home, away in (("A", "B"), ("C", "A"), ("D", "B")):
        assert rebuilt.predict_game(game(home, away)) == predictor.predict_game(
            game(home, away)
        )


def test_the_sides_survive_a_rebuild_with_different_anchors(
    game: GameFactory,
) -> None:
    """A consumer without the publisher's anchor file still reads the same
    offense: the release holds absolutes, and the prior is only how they are
    stored on the way in."""
    predictor = _predictor({"g": LOPSIDED}, anchors={"A": 1200})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    rebuilt = CompoundGlickoPredictor.from_ratings(
        "test_league", predictor.ratings, game_epa=EpaIndex(), anchors={}
    )

    assert rebuilt.get_units("A") == predictor.get_units("A")


def test_the_offseason_pulls_the_children_toward_their_prior(
    game: GameFactory,
) -> None:
    """Toward the anchor, at the default share -- the same place the parent goes."""
    predictor = _predictor({"g": LOPSIDED}, season_regression=0.5, anchors={"A": 1200})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    before = predictor.get_units("A").offense.rating

    predictor.pass_season()

    assert predictor.get_units("A").offense.rating == pytest.approx(
        1200 + (before - 1200) / 2
    )


def test_full_regression_forgets_the_children_entirely(game: GameFactory) -> None:
    predictor = _predictor({"g": LOPSIDED}, season_regression=1.0)
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    predictor.pass_season()

    assert predictor.unit_gap(game("A", "B")) == pytest.approx(0)


def test_deviations_grow_between_games_at_their_own_rate_and_cap_at_their_own_initial(
    game: GameFactory,
) -> None:
    """Each side on its own clock, so an offense can be the steadier of the two."""
    predictor = _predictor(
        {"g": LOPSIDED},
        offense_initial_rd=120,
        defense_initial_rd=180,
        offense_weekly_rd_increase=0,
        defense_weekly_rd_increase=50,
    )
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    after_game = predictor.get_units("A")

    predictor.pass_week()
    grown = predictor.get_units("A")
    assert grown.offense.rating_deviation == after_game.offense.rating_deviation
    assert grown.defense.rating_deviation > after_game.defense.rating_deviation

    for _ in range(20):
        predictor.pass_week()
    assert predictor.get_units("A").offense.rating_deviation < 120
    assert predictor.get_units("A").defense.rating_deviation == 180


def test_the_offseason_widens_each_side_by_its_own_increase(
    game: GameFactory,
) -> None:
    predictor = _predictor(
        {"g": LOPSIDED},
        offense_season_rd_increase=0,
        defense_season_rd_increase=30,
    )
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    before = predictor.get_units("A")

    predictor.pass_season()

    after = predictor.get_units("A")
    assert after.offense.rating_deviation == before.offense.rating_deviation
    assert after.defense.rating_deviation == pytest.approx(
        (before.defense.rating_deviation**2 + 30**2) ** 0.5
    )


def test_the_childrens_clock_defaults_to_the_parents() -> None:
    predictor = _predictor(initial_rd=333, weekly_rd_increase=7)
    assert (predictor._offense_initial_rd, predictor._defense_initial_rd) == (333, 333)
    assert predictor._offense_weekly_rd_increase == 7
    assert predictor._defense_weekly_rd_increase == 7
    assert predictor._offense_season_rd_increase == 120  # the parent's default
    assert predictor._defense_season_rd_increase == 120

    split = _predictor(initial_rd=333, offense_initial_rd=100)
    assert (split._offense_initial_rd, split._defense_initial_rd) == (100, 333)


def test_the_combination_is_by_precision(game: GameFactory) -> None:
    """The parent's `1 / rd^2` against what the units have earned.

    Checked against the formula rather than by direction, because the
    direction is what the linear blend also had and the formula is the
    point: the weight on the units is the precision they earned over their
    prior, times `unit_weight`, and nothing else.
    """
    predictor = _predictor({"g": LOPSIDED}, unit_weight=2.0)
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    parent = predictor.get_rating("A")
    parent_precision = 1 / parent.rating_deviation**2
    earned = 2.0 * predictor.unit_information("A")
    expected = (
        parent_precision * parent.rating + earned * predictor.unit_rating("A")
    ) / (parent_precision + earned)

    assert predictor._blended_rating("A") == pytest.approx(expected)
    assert earned > 0


def test_out_of_range_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        _predictor(unit_weight=-0.1)
    for parent_share in (-0.1, 1.5):
        with pytest.raises(ValueError):
            _predictor(parent_share=parent_share)
    with pytest.raises(ValueError):
        _predictor(offense_scale=0.0)
    with pytest.raises(ValueError):
        _predictor(defense_scale=-1.0)
    with pytest.raises(ValueError):
        _predictor(offense_initial_rd=0.0)
    with pytest.raises(ValueError):
        _predictor(defense_initial_rd=0.0)
    with pytest.raises(ValueError):
        _predictor(offense_weekly_rd_increase=-1.0)
    with pytest.raises(ValueError):
        _predictor(defense_season_rd_increase=-1.0)
    # 0 is a real increase -- the parent's own is 0 in ncaafb's fit.
    assert _predictor(offense_weekly_rd_increase=0.0)._offense_weekly_rd_increase == 0


def test_zero_is_the_off_switch_for_both_fractions() -> None:
    predictor = _predictor(unit_weight=0.0, parent_share=0.0)
    assert (predictor.unit_weight, predictor.parent_share) == (0.0, 0.0)


def test_priors_are_kept_under_this_class_name() -> None:
    """The parent's opponent priors are by model, and this is a different one."""
    manager = _predictor()._prior_manager
    assert isinstance(manager, OpponentPriorManager)
    assert "CompoundGlickoPredictor" in str(manager._prior_path)

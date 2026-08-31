from datetime import datetime, timezone

import pytest
from call_it_what_you_want import TeamClassification, TeamNamer
from endgame.types import Game, Season, Week

from division_anchors import (
    Fit,
    Tier,
    TierGame,
    _anchors,
    _Classifier,
    _drop_thin_conferences,
    _first_years,
    _tier_games,
    _tier_team_counts,
    _unplaceable_divisions,
    fit_tiers,
    main,
)


def _games(
    home: str, away: str, home_wins: int, away_wins: int, neutral: bool = True
) -> list[TierGame]:
    h, a = _tier(home), _tier(away)
    return [TierGame(h, a, 1.0, neutral)] * home_wins + [
        TierGame(h, a, 0.0, neutral)
    ] * away_wins


def _tier(name: str) -> Tier:
    """"fbs" is a bare division; "fbs/SEC" is a conference inside one."""
    division, _, conference = name.partition("/")
    return Tier(division, conference or None)


def test_a_tier_that_wins_more_rates_higher() -> None:
    fit = fit_tiers(_games("fbs", "fcs", 90, 10))

    assert fit.ratings[_tier("fbs")] > fit.ratings[_tier("fcs")]


def test_the_gap_matches_the_win_rate() -> None:
    """A 3:1 record is 10**(gap/400) = 3, so ~191 points."""
    fit = fit_tiers(_games("fbs", "fcs", 75, 25))

    gap = fit.ratings[_tier("fbs")] - fit.ratings[_tier("fcs")]
    assert gap == pytest.approx(191, abs=5)


def test_even_tiers_land_together_at_the_mean() -> None:
    fit = fit_tiers(_games("fbs", "fcs", 50, 50), mean=1500)

    assert fit.ratings[_tier("fbs")] == pytest.approx(1500, abs=1)
    assert fit.ratings[_tier("fcs")] == pytest.approx(1500, abs=1)


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
    gap = fit.ratings[_tier("fbs")] - fit.ratings[_tier("fcs")]
    assert gap == pytest.approx(0, abs=15)


def test_a_transitive_gap_is_recovered_without_direct_games() -> None:
    """D-III never plays FBS, so its scale has to come through FCS."""
    fit = fit_tiers(_games("fbs", "fcs", 75, 25) + _games("fcs", "d3", 75, 25))

    gap = fit.ratings[_tier("fbs")] - fit.ratings[_tier("d3")]
    assert gap == pytest.approx(382, abs=15)


def test_a_lopsided_gap_is_not_compressed_by_the_games_around_it() -> None:
    """The bug this fit is shaped to avoid.

    The tens of thousands of games played *inside* a division cancel out of
    the division gradient entirely, so a step scaled by the whole pool moves
    the ladder ~70x too slowly and stops early with everything bunched up.
    The gap here has to come back the same whether or not the pool is full
    of games that say nothing about it.
    """
    across = _games("fbs", "fcs", 90, 10)
    alone = fit_tiers(across)
    buried = fit_tiers(across + _games("fbs", "fbs", 5000, 5000))

    gap = alone.ratings[_tier("fbs")] - alone.ratings[_tier("fcs")]
    assert gap == pytest.approx(382, abs=10)
    buried_gap = buried.ratings[_tier("fbs")] - buried.ratings[_tier("fcs")]
    assert buried_gap == pytest.approx(gap, abs=10)


def test_a_conference_does_not_float_out_of_its_division() -> None:
    """The reason the fit is two-level rather than flat over conferences.

    d3/strong beats the rest of D-III handily, and never plays anyone
    outside it. Rated flat, nothing stops it climbing past FCS. Its level
    has to stay tied to the division term the cross-division games pin.
    """
    games = (
        _games("fbs", "fcs", 75, 25)
        + _games("fcs", "d3", 75, 25)
        + _games("d3/strong", "d3/weak", 90, 10)
        + _games("d3/weak", "d3", 50, 50)
        + _games("d3/strong", "d3", 50, 50)
    )

    fit = fit_tiers(games)

    assert fit.ratings[_tier("d3/strong")] > fit.ratings[_tier("d3/weak")]
    assert fit.ratings[_tier("d3/strong")] < fit.ratings[_tier("fcs")]


def test_conference_offsets_are_centred_inside_their_division() -> None:
    """Otherwise the split between division and offset means nothing."""
    games = _games("fbs", "fcs", 75, 25) + _games("fbs/a", "fbs/b", 75, 25)

    fit = fit_tiers(games)

    level = fit.divisions["fbs"]
    above = fit.ratings[_tier("fbs/a")] - level
    below = fit.ratings[_tier("fbs/b")] - level
    assert above > 0 > below
    assert above + below == pytest.approx(0, abs=5)


def test_no_games_is_not_a_crash() -> None:
    assert fit_tiers([]) == ({}, {}, 0.0)


def test_a_conference_that_only_plays_itself_falls_back_to_its_division() -> None:
    """The NESCAC case, and the reason the threshold counts crossing games.

    A round-robin conference racks up hundreds of games without ever
    playing anyone else. Every one of them cancels out of its own gradient,
    so counting the whole schedule passes a threshold on evidence that
    does not exist.
    """
    games = _games("d3/nescac", "d3/nescac", 600, 600) + _games("d3", "fcs", 20, 80)

    resolved, folded = _drop_thin_conferences(games)

    assert folded == {_tier("d3/nescac"): _tier("d3")}
    assert all(g.home_tier.conference is None for g in resolved)


def test_a_conference_that_plays_out_keeps_its_own_offset() -> None:
    games = _games("d3/open", "d3/other", 300, 300)

    _, folded = _drop_thin_conferences(games)

    assert folded == {}


def test_a_conference_with_too_few_teams_falls_back_to_its_division() -> None:
    """The one-team-tier case, which crossing games alone does not catch.

    A single team playing a full out-of-conference schedule clears the games
    threshold by itself, and the "conference" offset it earns is that one
    team's fitted level with nothing to average it against.
    """
    games = _games("d2/gulf_south", "d2/other", 300, 300)

    _, folded = _drop_thin_conferences(games, {_tier("d2/gulf_south"): 1})

    assert folded == {_tier("d2/gulf_south"): _tier("d2")}


def test_a_conference_with_enough_teams_keeps_its_own_offset() -> None:
    games = _games("d2/gulf_south", "d2/other", 300, 300)

    _, folded = _drop_thin_conferences(games, {_tier("d2/gulf_south"): 3})

    assert folded == {}


def test_team_counts_are_taken_over_every_season_played() -> None:
    """Three teams that were never in the conference at the same time still
    count as three: the fit pools all seasons, so its view of the tier does
    too."""
    classifier = _classifier(
        {
            ("Early", 2000): _found("d2", "gulf_south"),
            ("Middle", 2001): _found("d2", "gulf_south"),
            ("Late", 2002): _found("d2", "gulf_south"),
        }
    )

    counts = _tier_team_counts(
        {"Early": [2000], "Middle": [2001], "Late": [2002]}, classifier
    )

    assert counts[_tier("d2/gulf_south")] == 3


def test_a_team_that_moved_counts_in_both_tiers() -> None:
    classifier = _classifier(
        {
            ("Mover", 2000): _found("d2", "gulf_south"),
            ("Mover", 2001): _found("d2", "lone_star"),
        }
    )

    counts = _tier_team_counts({"Mover": [2000, 2001]}, classifier)

    assert counts[_tier("d2/gulf_south")] == 1
    assert counts[_tier("d2/lone_star")] == 1


def _found(division: str, conference: str | None = None) -> TeamClassification:
    """One recorded classification. The espn_id and league don't reach the
    code under test -- `_Classifier` reads them on the way *in*, and these
    tests start from the answer that lookup returns."""
    return TeamClassification("0", 0, "ncaafb", division, conference)


class _FakeClassifier(_Classifier):
    """A `_Classifier` reading a dict instead of the recorded ESPN data.

    Subclassed rather than monkeypatched so the seam under test is the one
    real callers use: everything above `_classification` -- resolving the
    lumped label, building a Tier -- is the shipped code.
    """

    def __init__(self, labels: dict[tuple[str, int], TeamClassification]) -> None:
        self._labels = labels
        self._resolved = {}

    def _classification(self, team: str, year: int) -> TeamClassification | None:
        return self._labels.get((team, year))


def _classifier(labels: dict[tuple[str, int], TeamClassification]) -> _Classifier:
    return _FakeClassifier(labels)


def test_the_lumped_label_is_filled_in_from_a_later_season() -> None:
    """ESPN lumped everything under FBS into one label through 2008.

    It is a season nobody recorded which division the team was in, not a
    division the team played in, so it is filled from the team's own later
    seasons rather than left to average D-II and D-III together.
    """
    classifier = _classifier(
        {
            ("Mount Union", 2002): _found("Division II/III", "Ohio"),
            ("Mount Union", 2015): _found("NCAA Division III", "Ohio"),
        }
    )
    classifier.resolve_lumped({"Mount Union": [2002, 2015]})

    assert classifier.tier("Mount Union", 2002) == Tier("NCAA Division III", "Ohio")


def test_a_team_that_moved_up_does_not_backfill_the_promotion() -> None:
    """The lumped label spans D-II and D-III, so only those can fill it.

    A program that appears lumped and is next seen in FCS was promoted;
    writing FCS onto its earlier seasons invents a move that had not
    happened yet.
    """
    classifier = _classifier(
        {
            ("Risen", 2004): _found("Division II/III", None),
            ("Risen", 2015): _found("FCS", "Big Sky"),
        }
    )
    classifier.resolve_lumped({"Risen": [2004, 2015]})

    assert classifier.tier("Risen", 2004) == Tier("Division II/III", None)
    assert classifier.tier("Risen", 2015) == Tier("FCS", "Big Sky")


def test_an_unclassified_team_gets_no_tier() -> None:
    """Guessing a division from the company a team keeps would be a rating
    decision dressed up as a data one."""
    classifier = _classifier({})

    assert classifier.tier("Nobody", 2010) is None


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
    classifier = _classifier(
        {("Big", 2024): _found("fbs", None), ("Small", 2024): _found("fcs", None)}
    )
    seasons = [_season(2024, _game(2024, completed=False))]

    assert _tier_games(seasons, TeamNamer.empty(), classifier) == []


def test_tier_games_reads_each_game_at_the_season_it_was_played() -> None:
    """A team that moved up is evidence about its new tier, not its old one.

    Reading every game through the team's *first* classification is what
    mixed the buckets together: a fifth of the FCS bucket is FBS by its
    last season, and those wins were being credited to FCS.
    """
    classifier = _classifier(
        {
            ("Big", 2004): _found("fcs", None),
            ("Big", 2024): _found("fbs", None),
            ("Small", 2004): _found("fcs", None),
            ("Small", 2024): _found("fcs", None),
        }
    )
    seasons = [_season(2004, _game(2004)), _season(2024, _game(2024))]

    tiers = [(g.home_tier, g.away_tier) for g in _tier_games(seasons, TeamNamer.empty(), classifier)]

    assert tiers == [(Tier("fcs"), Tier("fcs")), (Tier("fbs"), Tier("fcs"))]


def _classified_seasons() -> tuple[list[Season], _Classifier]:
    """Two teams over four seasons; one of them moves up in 2004."""
    seasons = [
        _season(year, _game(year, home="Risen", away="Steady"))
        for year in (2002, 2003, 2004, 2005)
    ]
    labels = {}
    for year in (2002, 2003, 2004, 2005):
        labels[("Steady", year)] = _found("d3")
        labels[("Risen", year)] = _found("d3" if year < 2004 else "fcs")
    return seasons, _classifier(labels)


def test_a_team_that_moves_up_gets_a_step_at_the_move() -> None:
    """The anchor a promoted program is judged against has to move with it."""
    seasons, classifier = _classified_seasons()
    fit = Fit({}, {"d3": 1200.0, "fcs": 1500.0}, 0.0)

    anchors = _anchors(seasons, TeamNamer.empty(), classifier, fit, {})

    assert anchors["Risen"] == [[2002, 1200.0], [2004, 1500.0]]


def test_a_team_that_never_moves_stays_a_bare_number() -> None:
    """Almost every team. A one-entry history would bury the ones that moved."""
    seasons, classifier = _classified_seasons()
    fit = Fit({}, {"d3": 1200.0, "fcs": 1500.0}, 0.0)

    anchors = _anchors(seasons, TeamNamer.empty(), classifier, fit, {})

    assert anchors["Steady"] == 1200.0


def test_a_season_the_fit_cannot_place_does_not_end_the_step() -> None:
    """The team kept playing; its last placed tier beats the league mean."""
    seasons = [
        _season(year, _game(year, home="Risen", away="Steady"))
        for year in (2002, 2003, 2004)
    ]
    classifier = _classifier(
        {
            ("Steady", 2002): _found("d3"),
            ("Steady", 2003): _found("d3"),
            ("Steady", 2004): _found("d3"),
            ("Risen", 2002): _found("d3"),
            # 2003 unclassified entirely
            ("Risen", 2004): _found("d3"),
        }
    )
    fit = Fit({}, {"d3": 1200.0}, 0.0)

    anchors = _anchors(seasons, TeamNamer.empty(), classifier, fit, {})

    assert anchors["Risen"] == 1200.0


def test_a_team_in_an_unplaceable_division_gets_no_anchor() -> None:
    """The all-star bowls: six squads that only ever play each other."""
    seasons = [_season(2014, _game(2014, home="East", away="West"))]
    classifier = _classifier(
        {("East", 2014): _found("All-star Bowls"), ("West", 2014): _found("All-star Bowls")}
    )
    fit = Fit({}, {"All-star Bowls": 1500.0}, 0.0)

    anchors = _anchors(
        seasons, TeamNamer.empty(), classifier, fit, {}, frozenset({"All-star Bowls"})
    )

    assert anchors == {}


def test_a_single_division_league_is_not_declared_unplaceable() -> None:
    """mens and womens are entirely D-I, so no game ever crosses a division.

    The guard is for a division that never connects to the *others*; with no
    others there is nothing to connect to. Reading that as "unplaceable"
    stripped the anchor off every team in the league and wrote an empty file.
    """
    games = [
        TierGame(_tier("d1/ACC"), _tier("d1/MEAC"), 1.0, True),
        TierGame(_tier("d1/Big Ten"), _tier("d1/ACC"), 0.0, True),
    ] * 500

    assert _unplaceable_divisions(games) == set()


def test_a_division_that_never_leaves_itself_is_unplaceable_when_others_exist() -> None:
    """The all-star bowls, which is what the guard is actually for."""
    games = _games("fbs", "fcs", 80, 20) + _games("bowls", "bowls", 3, 3)

    assert _unplaceable_divisions(games) == {"bowls"}

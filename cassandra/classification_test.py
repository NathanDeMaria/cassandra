from cassandra.classification import files_every_division, recorded_year


def test_a_season_that_crosses_new_year_is_filed_under_the_year_it_ends() -> None:
    assert recorded_year("mens", 2021) == 2022
    assert recorded_year("womens", 2021) == 2022


def test_a_season_inside_one_year_is_filed_under_that_year() -> None:
    assert recorded_year("ncaafb", 2021) == 2021


def test_only_basketball_files_every_division() -> None:
    assert files_every_division("mens")
    assert not files_every_division("ncaafb")

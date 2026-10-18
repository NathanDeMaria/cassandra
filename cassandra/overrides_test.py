from datetime import date

import pytest

from .overrides import Override, parse, split, teams

TEXT = """
[[qb_out]]
team = "UCLA Bruins"
until = 2026-11-28
note = "Iamaleava out for the season"

[[exclude]]
team = "Syracuse Orange"
until = 2026-10-17
"""


def test_overrides_parse_with_their_dates():
    assert parse(TEXT) == [
        Override(
            "qb_out", "UCLA Bruins", date(2026, 11, 28), "Iamaleava out for the season"
        ),
        Override("exclude", "Syracuse Orange", date(2026, 10, 17), ""),
    ]


def test_an_override_applies_through_its_until_date():
    active, expired = split(parse(TEXT), date(2026, 10, 18))
    assert teams(active, "qb_out") == ["UCLA Bruins"]
    assert teams(active, "exclude") == []
    assert teams(expired, "exclude") == ["Syracuse Orange"]
    active, _ = split(parse(TEXT), date(2026, 10, 17))
    assert teams(active, "exclude") == ["Syracuse Orange"]


@pytest.mark.parametrize(
    "text",
    [
        '[[qb_out]]\nteam = "UCLA Bruins"\n',  # no until
        "[[qb_out]]\nuntil = 2026-11-28\n",  # no team
        '[[qb_in]]\nteam = "UCLA Bruins"\nuntil = 2026-11-28\n',  # unknown kind
    ],
)
def test_a_malformed_override_is_an_error(text):
    with pytest.raises(ValueError):
        parse(text)

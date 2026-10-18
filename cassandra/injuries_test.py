from .injuries import (
    Roster,
    box_players,
    notes,
    opponents,
    passing,
    rosters,
    sentences,
)

UCLA = Roster(
    "UCLA",
    frozenset({"UCLA", "UCLA Bruins"}),
    frozenset({"Nico Iamaleava", "Wayne Knight"}),
    frozenset({"Nico Iamaleava", "Ty Dieffenbach"}),
)
OREGON = Roster(
    "Oregon",
    frozenset({"Oregon", "Oregon Ducks"}),
    frozenset({"Dante Moore", "Evan Stewart"}),
    frozenset({"Dylan Raiola"}),
)

STORY = """<p>No. 21 UCLA overcame an injury to starting quarterback
<a href="x">Nico Iamaleava</a> to remain undefeated.</p>
<p>Iamaleava did not return in the second half because of a lower leg injury.</p>
<p>Oregon's defense was dealing with injuries all night, Lanning said.</p>
<p>Evan Stewart (ankle) was limited after halftime.</p>
<p>Boise State took a knee on the final possession of the game.</p>"""


def test_sentences_strip_tags_and_split():
    found = sentences(
        "<p>One sentence here, long enough.</p><p>And a second one right here.</p>"
    )
    assert found == ["One sentence here, long enough.", "And a second one right here."]


def test_notes_tie_sentences_to_teams_and_flag_quarterbacks():
    found = notes(STORY, [UCLA, OREGON], "Recap")
    assert [(n.team, n.qb) for n in found] == [
        ("UCLA", True),
        ("UCLA", True),  # by the passer's surname alone
        ("Oregon", False),  # by the team's name
        ("Oregon", False),  # by a player, and "(ankle)"
    ]
    assert not any("knee" in n.sentence for n in found)


def test_rosters_pool_box_scores_by_espn_id():
    summary = {
        "boxscore": {
            "players": [
                {
                    "team": {"id": "26", "location": "UCLA"},
                    "statistics": [
                        {
                            "name": "passing",
                            "athletes": [
                                {"athlete": {"displayName": "Nico Iamaleava"}}
                            ],
                        },
                        {
                            "name": "rushing",
                            "athletes": [{"athlete": {"displayName": "Wayne Knight"}}],
                        },
                    ],
                },
                {
                    "team": {"id": "2483", "location": "Oregon"},
                    "statistics": [
                        {
                            "name": "passing",
                            "athletes": [{"athlete": {"displayName": " Team"}}],
                        }
                    ],
                },
            ]
        }
    }
    assert box_players(summary)["2483"] == ({"Oregon"}, set(), set())
    (ucla,) = rosters([summary, {}], {"26": "UCLA Bruins"})
    assert ucla.team == "UCLA Bruins"
    assert ucla.passers == {"Nico Iamaleava"}
    assert ucla.players == {"Nico Iamaleava", "Wayne Knight"}
    assert ucla.names == {"UCLA", "UCLA Bruins"}


def _box(espn_id: str, names: dict, passers: list[str]) -> dict:
    return {
        "team": {"id": espn_id, **names},
        "statistics": [
            {
                "name": "passing",
                "athletes": [{"athlete": {"displayName": p}} for p in passers],
            }
        ],
    }


def test_an_opponents_quarterback_is_the_opponents_news():
    # Boston College's recap, 2026-10-17: the injured QB was Pittsburgh's.
    summary = {
        "boxscore": {
            "players": [
                _box("103", {"location": "Boston College"}, ["Mason McKenzie"]),
                _box("221", {"location": "Pitt"}, ["Corey Dailey"]),
            ]
        }
    }
    teams = {"103": "Boston College Eagles"}
    (pitt,) = opponents([summary], teams)
    assert pitt.team == "Pitt" and pitt.passers == {"Corey Dailey"}
    story = (
        "<p>Making his first career start after QB Mason Heintschel was lost for "
        "the season with a torn ACL, true freshman Corey Dailey threw for 193.</p>"
    )
    (note,) = notes(story, rosters([summary], teams) + [pitt], "Recap")
    assert (note.team, note.qb) == ("Pitt", True)


def test_passing_reads_completions_and_attempts_most_attempts_first():
    summary = {
        "boxscore": {
            "players": [
                {
                    "team": {"id": "26"},
                    "statistics": [
                        {
                            "name": "passing",
                            "labels": ["C/ATT", "YDS"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Backup"},
                                    "stats": ["1/2", "9"],
                                },
                                {
                                    "athlete": {"displayName": "Starter"},
                                    "stats": ["24/31", "250"],
                                },
                                {
                                    "athlete": {"displayName": "Team"},
                                    "stats": ["0/0", "0"],
                                },
                            ],
                        }
                    ],
                },
                {"team": {"id": "2483"}, "statistics": []},
            ]
        }
    }
    assert passing(summary, "26") == [("Starter", 24, 31), ("Backup", 1, 2)]
    assert passing(summary, "2483") == []
    assert passing({}, "26") == []

"""Injury news on a pick's two teams, read out of ESPN's game stories.

ESPN's college football injury feed is empty (its newest entry is from
2022), and conference availability reports come out the day before a game,
not the Saturday night a pick is made. What there is on Saturday night is
the AP story on each team's last games -- the recap, and a preview of the
next game once one is up. Those name the injuries that mattered: a starter
who left in the third quarter, a quarterback "out for the rest of the
season", a backup making his first start.

`notes` pulls the sentences that read as injury news, and says which team
each is about from the players the game's box score lists for it. A
sentence naming a quarterback -- anyone with a pass in those box scores, or
the word itself -- is flagged, since that is the injury the model can
take: `picks.py --qb-out TEAM`. The rest are for reading: the model has no
input for a receiver or a corner.

This is a text match, so it over-reads ("dealing with injuries" is no
news). A team's story also covers its opponent's injuries; reading the
opponents' box scores too (`opponents`) ties those sentences to the
opponent, so they can be left out. It is a list to read before betting,
not a filter.
"""

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

_INJURY = re.compile(
    r"injur|concussion|\bprotocol\b|\bACL\b|\bMCL\b|\btorn\b|\btore\b|fractur|"
    r"\bbroken\b|sprain|carted|left the game|did not return|didn.t return|"
    r"ruled out|sidelined|out for the (?:season|year|rest)|season-ending|"
    r"surgery|\binactive\b|did not play|didn.t play|suspend|"
    r"\((?:ankle|knee|shoulder|hamstring|foot|wrist|hand|leg|arm|back|neck|"
    r"head|illness|undisclosed|lower body|upper body)\)",
    re.IGNORECASE,
)
_QB = re.compile(r"quarterback|\bQB\b")
_TAGS = re.compile(r"<[^>]+>")
_SENTENCE = re.compile(r"(?<=[.!?])[\"”’]?\s+|\n+")


class Roster(NamedTuple):
    """Who a team is in the stories: its names, its players, its passers."""

    team: str
    names: frozenset[str]
    players: frozenset[str]
    passers: frozenset[str]


class Note(NamedTuple):
    team: str | None
    qb: bool
    sentence: str
    source: str


def sentences(story: str) -> list[str]:
    """A story's HTML as plain sentences."""
    text = html.unescape(_TAGS.sub(" ", story or ""))
    found = (" ".join(part.split()) for part in _SENTENCE.split(text))
    return [s for s in found if len(s) > 20]


def _mentions(sentence: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", sentence) is not None


def _surname(name: str) -> str | None:
    parts = [
        p for p in name.split() if p.rstrip(".") not in ("Jr", "Sr", "II", "III", "IV")
    ]
    return parts[-1] if len(parts) > 1 and len(parts[-1]) > 3 else None


def notes(story: str, rosters: Sequence[Roster], source: str) -> list[Note]:
    """The injury sentences in one story, each tied to a team where it names one.

    A team is named by a player of its own, in full, or by a passer's
    surname, since stories call a quarterback by his surname after the
    first mention. Failing that, by the team's own name.
    """
    found = []
    for sentence in sentences(story):
        if not _INJURY.search(sentence):
            continue
        team, qb = None, bool(_QB.search(sentence))
        for roster in rosters:
            passer = any(
                _mentions(sentence, p)
                or ((s := _surname(p)) is not None and _mentions(sentence, s))
                for p in roster.passers
            )
            if passer or any(_mentions(sentence, p) for p in roster.players):
                team, qb = roster.team, qb or passer
                break
        if team is None:
            named = [
                r.team for r in rosters if any(_mentions(sentence, n) for n in r.names)
            ]
            team = named[0] if len(named) == 1 else None
        found.append(Note(team, qb, sentence, source))
    return found


def box_players(summary: Mapping) -> dict[str, tuple[set[str], set[str], set[str]]]:
    """Each ESPN team id in a game summary: its names, its players, its passers."""
    teams: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for side in (summary.get("boxscore") or {}).get("players") or []:
        team = side.get("team") or {}
        names: set[str] = {
            name
            for k in ("location", "shortDisplayName", "displayName")
            if (name := team.get(k))
        }
        players: set[str] = set()
        passers: set[str] = set()
        for category in side.get("statistics") or []:
            for athlete in category.get("athletes") or []:
                name = ((athlete.get("athlete") or {}).get("displayName") or "").strip()
                if not name or name == "Team":
                    continue
                players.add(name)
                if category.get("name") == "passing":
                    passers.add(name)
        teams[str(team.get("id"))] = (names, players, passers)
    return teams


def passing(summary: Mapping, espn_id: str) -> list[tuple[str, int, int]]:
    """Who threw for a team in a game: (name, completions, attempts), most attempts first.

    Who started at quarterback last week is the injury check a recap can't
    be counted on for: a starter missing from the box score is news even
    when no sentence says so.
    """
    found = []
    for side in (summary.get("boxscore") or {}).get("players") or []:
        if str((side.get("team") or {}).get("id")) != str(espn_id):
            continue
        for category in side.get("statistics") or []:
            if category.get("name") != "passing":
                continue
            labels = category.get("labels") or []
            at = labels.index("C/ATT") if "C/ATT" in labels else 0
            for athlete in category.get("athletes") or []:
                name = ((athlete.get("athlete") or {}).get("displayName") or "").strip()
                stats = athlete.get("stats") or []
                made, _, tried = (stats[at] if len(stats) > at else "").partition("/")
                if name and name != "Team" and made.isdigit() and tried.isdigit():
                    found.append((name, int(made), int(tried)))
    return sorted(found, key=lambda p: -p[2])


def rosters(summaries: Iterable[Mapping], teams: Mapping[str, str]) -> list[Roster]:
    """`teams` (ESPN id → name) as each is in a set of game summaries."""
    pooled: dict[str, tuple[set[str], set[str], set[str]]] = {
        espn_id: (set(), set(), set()) for espn_id in teams
    }
    for summary in summaries:
        for espn_id, found in box_players(summary).items():
            if espn_id in pooled:
                for into, more in zip(pooled[espn_id], found):
                    into |= more
    return [
        Roster(
            teams[espn_id],
            frozenset(names | {teams[espn_id]}),
            frozenset(players),
            frozenset(passers),
        )
        for espn_id, (names, players, passers) in pooled.items()
    ]


def opponents(summaries: Iterable[Mapping], teams: Mapping[str, str]) -> list[Roster]:
    """Every other team in a set of game summaries, by the name its box score uses.

    A team's recap is as much about its opponent: "making his first career
    start after QB Mason Heintschel was lost for the season" in Boston
    College's recap was about Pittsburgh. With the opponents' rosters in
    `notes`, a sentence like that is tied to the team it's about rather
    than read as news of no team -- or of the pick's.
    """
    pooled: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for summary in summaries:
        for espn_id, found in box_players(summary).items():
            if espn_id in teams:
                continue
            into = pooled.setdefault(espn_id, (set(), set(), set()))
            for have, more in zip(into, found):
                have |= more
    return [
        Roster(
            max(names, key=len),
            frozenset(names),
            frozenset(players),
            frozenset(passers),
        )
        for names, players, passers in pooled.values()
        if names
    ]


def story(summary: Mapping) -> tuple[str, str] | None:
    """A summary's AP story and what it is: ("Recap", html) or ("Preview", html)."""
    article = summary.get("article") or {}
    if not article.get("story"):
        return None
    return article.get("type") or "Story", article["story"]

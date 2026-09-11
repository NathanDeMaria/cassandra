"""Who started at quarterback, and whether he took a snap, from the play text.

`PLAY_SCHEMA` carries no player fields at all -- team ids, drive bookkeeping
and a free-text description -- so the only place a quarterback's name appears
is the sentence ESPN writes for a play. Two things are read out of it:

- **who started**: the passer with the most attempts for a team in a game.
- **who took a snap**: everyone credited with a pass or a rush.

The second is what makes an availability flag possible. A team's expected
starter is whoever started its previous game; if that player records no pass
and no rush in this one, he did not play.

## Three formats

ncaafb play text arrives in three shapes, and all of them have to be read or
a quarterback looks absent when he was only described differently:

    Taylen Green pass complete to O'Mega Blake for 16 yds
    (05:55) Shotgun Nussmeier,Garrett pass incomplete deep left to Hilton Jr.,Chris
    (15:00) No Huddle-Shotgun #7 K.Jackson pass complete short left to #3 C.Brown

Which one dominates changes by season, which is the trap. The first covers
almost all of 2025 and the third almost none of it -- and in 2026 the third
is 20,387 of 22,692 plays in a single week. A parser validated on one week
of one season and shipped is a parser that silently stops working, and this
one did: before the `#7 K.Jackson` shape was handled, the 2026 index was
empty and the feature did nothing at all for the season being played.

They write names three ways -- `First Last`, `Last,First`, `F.Last` -- so
nothing matches across them without normalizing, hence `name_key`, which
reduces all three to a last name and a first initial.

The known limitation is a multi-word surname: "Michael Van Buren Jr." and
"Van Buren Jr.,Michael" normalize differently, because the first-format
parse cannot tell which of the middle tokens belong to the surname, and the
third format is read only up to the first space after the initial. Those
players get more than one key and can read as a quarterback change or an
absence that did not happen. It is a handful of players and it is why the
availability index is a naive one.

A kneel-down is deliberately not a snap here. It is neither a pass nor a
rush in the box score, which is the definition this is built to, and a
quarterback who came back on to kneel out a half is not evidence that he
was available.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import NamedTuple

#: Suffixes that are not part of a surname for matching purposes.
_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"})

#: `F.Last`, the third format's name shape.
_INITIAL = re.compile(r"^[A-Z]\.[^ ]")

#: `First Last pass complete|incomplete ...`, the common format. Bounded
#: rather than greedy: the sentence carries a second name after "to", and an
#: unbounded prefix would swallow a preceding clause on a play whose text
#: runs two sentences together.
_PASSER_PLAIN = re.compile(
    r"^([A-Z][^,;#]{1,39}?) pass (?:complete|incomplete|intercepted)"
)

#: `First Last run for ...` and `First Last sacked by ...`. A sack is a snap
#: the quarterback took, so it counts for availability even though it is not
#: a rush in the box score.
_RUSHER_PLAIN = re.compile(r"^([A-Z][^,;]{1,39}?) (?:run for|sacked by)")

#: The clock the second format opens with, and the formations that follow it.
#: A closed set, read off the data rather than guessed: three spellings cover
#: every prefixed play, and the surname behind them can itself contain spaces
#: ("Del Rio-Wilson,Angel"), so the prefix has to come off by name rather
#: than by counting tokens.
_CLOCK = re.compile(r"^\(\d+:\d+\)\s*")
_FORMATION = re.compile(r"^(?:No Huddle-Shotgun|No Huddle|Shotgun)\s*")

#: `#N F.Last pass|rush|sacked ...`, the third format, once the prefix is
#: off. The jersey number is what tells it from the other two. Bounded to
#: a single token after the initial so the verb can never be swallowed.
_PASSER_HASH = re.compile(
    r"^#\d+ ([A-Z]\.[\w'\-]+) pass (?:complete|incomplete|intercepted)"
)
_CARRIER_HASH = re.compile(r"^#\d+ ([A-Z]\.[\w'\-]+) (?:pass|rush|sacked)\b")

#: `Last,First pass|rush|sacked ...` once the prefix is off.
_CARRIER_COMMA = re.compile(
    r"^([\w'\-. ]{1,30},[\w'\-. ]{1,25}?) (?:pass|rush|sacked)\b"
)
_PASSER_COMMA = re.compile(r"^([\w'\-. ]{1,30},[\w'\-. ]{1,25}?) pass\b")


def _unprefixed(text: str) -> str:
    """The second format with its clock and formation removed.

    A no-op on the common format, which starts with the name.
    """
    return _FORMATION.sub("", _CLOCK.sub("", text))


def name_key(name: str) -> str:
    """A name reduced to something that matches across both text formats.

    Last name plus first initial, lowercased, suffixes dropped. Crude on
    purpose: it has to survive `Garrett Nussmeier` and `Nussmeier,Garrett`
    being the same person, and it is not trying to be an identity.
    """
    name = name.strip()
    if "," in name:
        last, _, first = name.partition(",")
    elif _INITIAL.match(name):
        # `K.Jackson`: the initial is the first name and everything after the
        # dot is the surname.
        first, last = name[0], name[2:]
    else:
        parts = [p for p in name.split() if p.lower().strip(".") not in _SUFFIXES]
        if not parts:
            return name.lower()
        first, last = parts[0], parts[-1]
    last = " ".join(
        p for p in last.split() if p.lower().strip(".") not in _SUFFIXES
    ).lower()
    first = first.strip()
    return f"{last} {first[:1].lower()}" if first else last


def _match(text: str, *patterns: re.Pattern[str]) -> str | None:
    for pattern in patterns:
        found = pattern.match(text)
        if found:
            return found.group(1).strip()
    return None


def passer(text: str | None) -> str | None:
    """The name credited with the pass, or None if this isn't a pass play."""
    if not text:
        return None
    return _match(
        _unprefixed(text.strip()), _PASSER_HASH, _PASSER_PLAIN, _PASSER_COMMA
    )


def ball_carrier(text: str | None) -> str | None:
    """Whoever passed, ran or was sacked on this play, if anyone named was.

    A sack counts: the quarterback took the snap, which is the question
    availability asks. It is not a rushing attempt and nothing here pretends
    it is.
    """
    if not text:
        return None
    return _match(
        _unprefixed(text.strip()),
        _CARRIER_HASH,
        _PASSER_PLAIN,
        _RUSHER_PLAIN,
        _CARRIER_COMMA,
    )


class TeamGameQb(NamedTuple):
    """One team's quarterback situation in one game.

    `starter` is the display name from whichever format produced it, so it
    reads like a person; `starter_key` is what matching is done on.
    `share` is the busiest passer's fraction of the team's attempts -- 30 of
    32 is a starter, 9 of 17 is a quarterback controversy or a parsing
    failure, and the two should not be read the same way.
    """

    starter: str
    starter_key: str
    attempts: int
    share: float
    snap_keys: frozenset[str]


def team_games(
    game_ids: Iterable[str],
    offense_team_ids: Iterable[str | None],
    texts: Iterable[str | None],
) -> dict[tuple[str, str], TeamGameQb]:
    """Per (game_id, team_id): who started, and who took a snap.

    Columns rather than row objects because the plays arrive as a pyarrow
    table and a week of them is twenty thousand rows.

    A team with no pass attempts is absent rather than present with an empty
    name -- "we don't know" is not a quarterback -- even if somebody ran the
    ball for it.
    """
    passers: dict[tuple[str, str], Counter[str]] = {}
    names: dict[tuple[str, str], dict[str, str]] = {}
    snaps: dict[tuple[str, str], set[str]] = {}
    for game_id, team_id, text in zip(game_ids, offense_team_ids, texts):
        if team_id is None:
            continue
        key = (str(game_id), str(team_id))
        carrier = ball_carrier(text)
        if carrier is not None:
            snaps.setdefault(key, set()).add(name_key(carrier))
        thrower = passer(text)
        if thrower is None:
            continue
        thrower_key = name_key(thrower)
        passers.setdefault(key, Counter())[thrower_key] += 1
        names.setdefault(key, {}).setdefault(thrower_key, thrower)

    out: dict[tuple[str, str], TeamGameQb] = {}
    for key, tally in passers.items():
        # Ties broken by key so a re-run gives the same answer.
        best = max(sorted(tally), key=lambda k: tally[k])
        total = sum(tally.values())
        out[key] = TeamGameQb(
            starter=names[key][best],
            starter_key=best,
            attempts=total,
            share=tally[best] / total,
            snap_keys=frozenset(snaps.get(key, ())),
        )
    return out


def starters(
    game_ids: Iterable[str],
    offense_team_ids: Iterable[str | None],
    texts: Iterable[str | None],
) -> dict[tuple[str, str], str]:
    """Likely starting quarterback per (game_id, team_id), by display name."""
    return {k: v.starter for k, v in team_games(game_ids, offense_team_ids, texts).items()}


def attempt_share(tally: Mapping[str, int]) -> float:
    """What fraction of a team's attempts the busiest passer threw."""
    total = sum(tally.values())
    return max(tally.values()) / total if total else 0.0

"""Who started at quarterback, and whether he took a snap, from the play text.

`PLAY_SCHEMA` carries no player fields at all -- team ids, drive bookkeeping
and a free-text description -- so the only place a quarterback's name appears
is the sentence ESPN writes for a play. Two things are read out of it:

- **who started**: the passer who threw a team's early attempts in a game.
- **who took a snap**: everyone credited with a pass or a rush.

The second is what makes an availability flag possible. If the team's
starter records no pass and no rush in a game, he did not play. Who the
starter is -- and when a man who has been covering for him becomes the
starter instead -- is `cassandra.qb_out_build`'s call, which sees the
season in order.

## Four formats

ncaafb play text arrives in three shapes, and all of them have to be read or
a quarterback looks absent when he was only described differently:

    Taylen Green pass complete to O'Mega Blake for 16 yds
    (05:55) Shotgun Nussmeier,Garrett pass incomplete deep left to Hilton Jr.,Chris
    (15:00) No Huddle-Shotgun #7 K.Jackson pass complete short left to #3 C.Brown
    (Shotgun) D.Gabriel pass short middle to D.Njoku to CLV 47 for 12 yards

Which one dominates changes by season, which is the trap. The first covers
almost all of 2025 and the third almost none of it -- and in 2026 the third
is 20,387 of 22,692 plays in a single week. A parser validated on one week
of one season and shipped is a parser that silently stops working, and this
one did: before the `#7 K.Jackson` shape was handled, the 2026 index was
empty and the feature did nothing at all for the season being played.

The fourth is the NFL feed, and it is the odd one: a completion carries no
word for itself at all, and a carry is `up the middle` rather than any kind
of "rush". Handled here because `QB_LEAGUES` covers nfl, and an index built
for a league whose text nobody parsed is a parameter that searches nothing.

They write names three ways -- `First Last`, `Last,First`, `F.Last` -- so
nothing matches across them without normalizing, hence `name_key`, which
reduces all of them to a last name and a first initial.

A name in the `F.Last` formats can run past one token -- `D.Williams Jr.`,
`M.Van Buren Jr.`, `C.Del Rio-Wilson`, `K.Ah Yat`, `A.St. Louis` -- so the
patterns are anchored on the verb, not on a space after the surname. The
first cut of this stopped at the space, and the cost was not "an extra key":
those quarterbacks matched nothing at all, their teams' starter became
whichever receiver threw one trick-play pass, and fifteen teams in 2025 --
Washington, LSU, James Madison, Marshall among them -- read as missing a
quarterback every week from the point that format took over.

The multi-word surname is still the soft spot across formats. `name_key`
folds a run of surname particles ("Van", "Del", "Ah", "St.") into the
surname so "Michael Van Buren Jr.", "Van Buren Jr.,Michael" and "M.Van
Buren Jr." agree; a two-word first name it cannot tell from a two-word
surname, and "Eddie Lee Marburger" and "E.Marburger" still differ. That
only bites at a boundary between formats, which is a week or two a season.

## Who started

The starter is not the busiest passer. A quarterback hurt in the first
quarter throws six passes and his backup throws thirty, and reading the
backup as the starter turns the next week's flag inside out: the injured
man's absence is expected, and his return reads as the backup going
missing. So attempts are weighted by when they happened -- each one counts
`STARTER_DECAY` of the one before it -- which makes the first few decisive
without being a hard rule. At 0.75 a man who threw the first three passes
outweighs everyone who threw after him; a man who threw only the first two
does not. `share` still reports the plain attempt fraction, so an early
injury shows up as a starter with a small share rather than being hidden.

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

#: Tokens that belong to the surname that follows them, read off the 2025
#: quarterbacks the one-token surname lost: Van Buren, Del Rio-Wilson, Ah
#: Yat, St. Louis, Vander Haar.
_PARTICLES = frozenset(
    {
        "van",
        "vander",
        "von",
        "de",
        "del",
        "della",
        "der",
        "di",
        "da",
        "la",
        "le",
        "st",
        "ah",
    }
)

#: How much less each pass attempt counts than the one before it when
#: deciding who started. See "Who started" above.
STARTER_DECAY = 0.75

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

#: What a play's text can open with before the name: a clock or a formation
#: in parentheses -- `(05:55)`, `(Shotgun)`, `(No Huddle, Shotgun)` -- or a
#: bare formation. Both can appear, in that order.
#:
#: The bare spellings are a closed set read off the data rather than guessed,
#: because the surname behind them can itself contain spaces
#: ("Del Rio-Wilson,Angel") and the prefix has to come off by name rather
#: than by counting tokens. The parenthesised form needs no such list.
_PARENTHESISED = re.compile(r"^\([^)]*\)\s*")
_FORMATION = re.compile(r"^(?:No Huddle-Shotgun|No Huddle|Shotgun)\s*")

#: An `F.Last` name: the initial, the first surname token, then up to three
#: more capitalised tokens -- `Jr.`, `III`, `Buren Jr.`, `Rio-Wilson` --
#: taken lazily, so the verb that follows is the stop and can never be
#: swallowed. Capitalised is the discriminator: every verb these patterns
#: look for is lowercase, and a jersey number or a yardage isn't a letter.
_INITIAL_NAME = r"[A-Z]\.[\w'\-.]+(?: [A-Z][\w'\-.]*){0,3}?"

#: `#N F.Last pass|rush|sacked ...`, the third format, once the prefix is
#: off. The jersey number is what tells it from the other two.
_PASSER_HASH = re.compile(
    rf"^#\d+ ({_INITIAL_NAME}) pass (?:complete|incomplete|intercepted)"
)
_CARRIER_HASH = re.compile(rf"^#\d+ ({_INITIAL_NAME}) (?:pass|rush|sacked)\b")

#: The NFL feed, which names everyone `F.Last` and writes a completion with
#: no verb for it at all -- `D.Gabriel pass short middle to D.Njoku` -- so
#: this cannot ask for "complete" the way the college patterns do. Anchored
#: on the initial-and-dot shape instead, which is what keeps it from reading
#: `Jaden Reddell 14 Yd pass from Gunner Stockton` -- a scoring line whose
#: first name is the *receiver* -- as a pass by Jaden Reddell.
#:
#: Its rushes have no rush verb either: a carry is `up the middle`, `left
#: end`, `right tackle`. `scrambles` and `sacked` are snaps the quarterback
#: took and count; `kicks`, `punts`, `kneels`, `spiked` and `reported` are
#: not carries and deliberately do not.
_PASSER_NFL = re.compile(rf"^({_INITIAL_NAME}) pass\b")
_CARRIER_NFL = re.compile(
    rf"^({_INITIAL_NAME}) (?:pass|sacked|scrambles|up the|left|right)\b"
)

#: `Last,First pass|rush|sacked ...` once the prefix is off.
_CARRIER_COMMA = re.compile(
    r"^([\w'\-. ]{1,30},[\w'\-. ]{1,25}?) (?:pass|rush|sacked)\b"
)
_PASSER_COMMA = re.compile(r"^([\w'\-. ]{1,30},[\w'\-. ]{1,25}?) pass\b")


def _unprefixed(text: str) -> str:
    """`text` with any leading clock and formation removed.

    Looped because the two stack: ncaafb writes `(15:00) No Huddle-Shotgun
    #7 K.Jackson ...`, a clock and then a bare formation. A no-op on the
    format that starts with the name.
    """
    for _ in range(4):
        stripped = _FORMATION.sub("", _PARENTHESISED.sub("", text))
        if stripped == text:
            return text
        text = stripped
    return text


def _is_suffix(token: str) -> bool:
    return token.lower().strip(".") in _SUFFIXES


def name_key(name: str) -> str:
    """A name reduced to something that matches across the text formats.

    Last name plus first initial, lowercased, suffixes dropped. Crude on
    purpose: it has to survive `Garrett Nussmeier`, `Nussmeier,Garrett` and
    `G.Nussmeier` being the same person, and it is not trying to be an
    identity.
    """
    name = name.strip()
    if "," in name:
        # `Nussmeier,Garrett`; occasionally `Emanuel, Jr.,Bert`, with the
        # suffix as a segment of its own.
        segments = [s.strip() for s in name.split(",")]
        segments = [s for s in segments if s and not _is_suffix(s)]
        last, first = segments[0], (segments[1] if len(segments) > 1 else "")
    elif _INITIAL.match(name):
        # `K.Jackson`: the initial is the first name and everything after the
        # dot is the surname.
        first, last = name[0], name[2:]
    else:
        parts = name.split()
        # Suffixes come off the tail only: "JR Wilson" is a first name.
        while len(parts) > 1 and _is_suffix(parts[-1]):
            parts.pop()
        if len(parts) == 1:
            # A bare surname -- "Ekleinpeter Jr." -- has no initial to add.
            return parts[0].lower()
        first, parts = parts[0], parts[1:]
        # Pull surname particles back onto the last token, so "Michael Van
        # Buren" gives "Van Buren" and "Carlos Del Rio-Wilson" gives "Del
        # Rio-Wilson", the same as the other two formats read them.
        start = len(parts) - 1
        while start > 0 and parts[start - 1].lower().strip(".") in _PARTICLES:
            start -= 1
        last = " ".join(parts[max(start, 0) :])
    last = " ".join(p for p in last.split() if not _is_suffix(p)).lower()
    first = first.strip()
    return f"{last} {first[:1].lower()}" if first else last


#: Not a person: some feeds credit a throwaway to the team itself.
_NOBODY = frozenset({"TEAM", "Team"})


def _match(text: str, *patterns: re.Pattern[str]) -> str | None:
    for pattern in patterns:
        found = pattern.match(text)
        if found:
            name = found.group(1).strip()
            return None if name in _NOBODY else name
    return None


def passer(text: str | None) -> str | None:
    """The name credited with the pass, or None if this isn't a pass play."""
    if not text:
        return None
    return _match(
        _unprefixed(text.strip()),
        _PASSER_HASH,
        _PASSER_NFL,
        _PASSER_PLAIN,
        _PASSER_COMMA,
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
        _CARRIER_NFL,
        _PASSER_PLAIN,
        _RUSHER_PLAIN,
        _CARRIER_COMMA,
    )


class TeamGameQb(NamedTuple):
    """One team's quarterback situation in one game.

    `starter` is the display name from whichever format produced it, so it
    reads like a person; `starter_key` is what matching is done on.
    `share` is the starter's fraction of the team's attempts -- 30 of 32 is
    a starter who played the game, 6 of 36 is one who left it early, 9 of 17
    is a quarterback controversy or a parsing failure, and none of those
    should be read the same way.
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
    table and a week of them is twenty thousand rows. They must come in game
    order, because the starter is decided by who threw *early*, not by who
    threw most (see "Who started" above).

    A team with no pass attempts is absent rather than present with an empty
    name -- "we don't know" is not a quarterback -- even if somebody ran the
    ball for it.
    """
    passers: dict[tuple[str, str], Counter[str]] = {}
    weighted: dict[tuple[str, str], Counter[str]] = {}
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
        tally = passers.setdefault(key, Counter())
        weighted.setdefault(key, Counter())[thrower_key] += STARTER_DECAY ** sum(
            tally.values()
        )
        tally[thrower_key] += 1
        names.setdefault(key, {}).setdefault(thrower_key, thrower)

    out: dict[tuple[str, str], TeamGameQb] = {}
    for key, tally in passers.items():
        # Ties broken by key so a re-run gives the same answer.
        best = max(sorted(tally), key=lambda k: weighted[key][k])
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
    return {
        k: v.starter for k, v in team_games(game_ids, offense_team_ids, texts).items()
    }


def attempt_share(tally: Mapping[str, int]) -> float:
    """What fraction of a team's attempts the busiest passer threw."""
    total = sum(tally.values())
    return max(tally.values()) / total if total else 0.0

"""Who probably started at quarterback, read off the play text.

Naive on purpose. `PLAY_SCHEMA` carries no player fields at all -- team ids,
drive bookkeeping and a free-text description -- so the only place a
quarterback's name appears is the sentence ESPN writes for a pass:

    Taylen Green pass complete to Rohan Jones for 62 yds for a TD
    Brendon Lewis pass incomplete

The name before "pass complete"/"pass incomplete" is the passer, and the
passer with the most attempts for a team in a game is that team's likely
starter. That is the whole model. It gets a trick play wrong, it cannot see
a quarterback who was hurt on the first series, and it has no opinion at all
about a team that threw nothing.

What it is for: asking whether the residuals know about quarterback changes
before anything is built to predict them. `cassandra.residuals` is explicit
that the ordering is measure-then-build, and this is the cheapest thing that
turns "add a QB adjustment" into a measurable axis. It is not a rating, it
is not predictive on its own -- a starter is only known *after* the game
whose plays name him -- and nothing in the predictor imports it.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping

#: The passer is whatever precedes the verb. Bounded rather than greedy: the
#: sentence can carry a second name ("pass complete to ..."), and an
#: unbounded prefix would swallow a preceding clause on the rare play whose
#: text runs two sentences together.
#:
#: The length bound is what keeps a malformed line from producing a
#: forty-word "name"; real ones run to about 25 characters ("Cortez Braham
#: Jr."), so 40 is loose enough to be uninteresting.
_PASSER = re.compile(r"^([A-Z][^,;]{1,39}?) pass (?:complete|incomplete)")


def passer(text: str | None) -> str | None:
    """The name credited with the pass, or None if this isn't a pass play."""
    if not text:
        return None
    found = _PASSER.match(text.strip())
    return found.group(1).strip() if found else None


def starters(
    game_ids: Iterable[str],
    offense_team_ids: Iterable[str | None],
    texts: Iterable[str | None],
) -> dict[tuple[str, str], str]:
    """Likely starting quarterback per (game_id, team_id).

    Columns rather than row objects because the plays arrive as a pyarrow
    table and a week of them is twenty thousand rows; zipping three lists
    beats building an object per play.

    Most attempts wins, ties broken by name so a re-run gives the same
    answer. A team with no pass attempts is absent rather than present with
    an empty string -- "we don't know" is not a quarterback.
    """
    counts: dict[tuple[str, str], Counter[str]] = {}
    for game_id, team_id, text in zip(game_ids, offense_team_ids, texts):
        if team_id is None:
            continue
        name = passer(text)
        if name is None:
            continue
        counts.setdefault((str(game_id), str(team_id)), Counter())[name] += 1
    return {
        key: max(sorted(tally), key=lambda n: tally[n]) for key, tally in counts.items()
    }


def attempt_share(tally: Mapping[str, int]) -> float:
    """What fraction of a team's attempts the busiest passer threw.

    The confidence reading for one game: a starter who threw 30 of 32 is a
    starter, and one who threw 9 of 17 is a quarterback controversy or a
    parsing failure, and the two should not be read the same way.
    """
    total = sum(tally.values())
    return max(tally.values()) / total if total else 0.0

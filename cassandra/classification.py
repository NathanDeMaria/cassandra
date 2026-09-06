"""ESPN's division labels, and the one of them that isn't a division.

`call_it_what_you_want` records which division and conference ESPN filed a
team under, per season. One label needs interpreting before anything can use
it, and two callers need the same interpretation: `division_anchors.py`,
which fits a rating ladder over the tiers, and `cassandra.residuals`, which
slices a model's error by them. A diagnostic that put a game in a different
tier from the fit it is diagnosing would be comparing two different leagues.
"""

from collections.abc import Callable, Mapping, Sequence

from call_it_what_you_want import TeamClassification

#: ESPN filed everything below FCS under one label through 2008 and split it
#: in two from 2011, so this is not a division a team played in -- it is a
#: season where nobody recorded which of the two it was.
LUMPED_DIVISION = "Division II/III"

#: The two the label above spans, and so the only two it can be filled in as.
SPANNED_DIVISIONS = frozenset({"NCAA Division II", "NCAA Division III"})


def resolve_spanning_label(
    teams: Mapping[str, Sequence[int]],
    classification: Callable[[str, int], TeamClassification | None],
) -> dict[str, str]:
    """Per team, which division the spanning label meant, where that is known.

    Filled in only from a season where ESPN recorded one of the two the label
    spans. A program that appears lumped and is next seen in FCS moved up;
    backfilling FCS onto its earlier seasons would be inventing a promotion
    that hadn't happened yet, so those seasons keep the spanning label and
    are handled as their own thing.

    The seasons a team played are asked first, then the latest season anybody
    played. ESPN re-surveyed everything below FCS in 2011, and a team is
    never asked about a year it has no game in -- which is how a program that
    had already folded keeps the label for every season it did play. Five
    ncaafb teams are exactly that.

    A team that moved between its last game and the survey is filled in from
    where it ended rather than where it was. That is the same trade already
    made inside the span, and it is worth those five teams.
    """
    resolved: dict[str, str] = {}
    last_recorded = max((year for years in teams.values() for year in years), default=0)
    for team, years in teams.items():
        for year in (*years, last_recorded):
            found = classification(team, year)
            if found is not None and found.division in SPANNED_DIVISIONS:
                resolved[team] = found.division
                break
    return resolved

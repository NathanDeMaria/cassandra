"""Games against opponents that aren't programs, left out of the replay.

An ncaafb season pickle is everything ESPN's scoreboard had, and the
scoreboard has games nobody would rate: a D-II school's tune-up against a
Bible college, a first-year program's scrimmage, an NAIA opponent that
appears once and never again. The registry files none of these under an NCAA
tier, so they enter the replay at the unfiled prior and lose by 27 points more
than even that predicts (measured on ncaafb `glicko_full`, own-side residual
of the 72 teams with five or fewer career games: -26.6 over 138 games).

Two things are wrong with rating those games. The opponent's rating is
credited for a result against a team the model has never had a chance to
measure -- Lenoir-Rhyne 92, University of Faith 0 -- and the team itself
ends up in a release with a rating nobody should read. `GlickoPredictor.
unanchored_prior` fixes most of the first; this fixes both by not rating the
game at all.

The rule
--------

A game is an exhibition when either side is **unanchored and has five or
fewer games in the league's whole stored history**, fixtures included.
Both halves matter:

- *Unanchored*, because the registry classifies every FBS and FCS program
  and most of D-II and D-III. A team with an anchor is a program, however
  few games the pickle has for it -- a new D-III school in its first season
  is filed, and rated.
- *Five career games*, because that is what tells a scrimmage opponent from
  an unfiled program. Point University has 33, Wayland Baptist 30, Waldorf
  24: NAIA programs that play NCAA schedules year after year, real enough to
  rate even without a tier. University of God's Chosen Disciples has one.
  Career rather than per-season because a league with no anchors at all --
  nfl -- would otherwise lose its opening week to a rule about small
  counts, and because the pickle for a season in progress carries the whole
  schedule as fixtures, so a real new program shows up with ten games from
  its first Saturday.

The threshold is a constant, not a knob. It is a data-cleaning decision --
which games are NCAA football -- and searching it against brier would tune
it to whatever helps the model rather than to what the games are. Five is
where the count separates cleanly: 72 teams at or under it, and the next
tier (6 to 19 games) is the NAIA programs above.

What it costs
-------------

0.18% of ncaafb's games. Nothing measurable on the remaining ones: the
opponents' residuals in the game after a scrimmage were +1.9 +/- 2.1 over
137 team-games, which is a sample too small to say the credit was hurting
anybody. The gain is hygiene -- ratings that were never earned don't get
published, and a first-year program's first five games don't teach the
model a rating for it until it is either filed or shows up for a sixth.

That last is the one real trade. A genuinely new NCAA program that the
registry hasn't filed yet is unrated for its first five games. Filing it
is the fix, and `call_it_what_you_want` is where that happens.

Where it runs
-------------

On the seasons as read from s3, before any model sees them --
`save_predictions.read_league` and the scripts that read seasons for a
rating replay. Not inside `generate_predictions`, whose tests replay two
teams for a handful of games and are exactly what the rule would drop, and
not in the index builders, which key on game ids and lose nothing by
knowing about a game nobody rated.
"""

from collections import Counter
from collections.abc import Iterable, Sequence

from call_it_what_you_want import TeamNamer
from endgame.types import Season

from .predictor.base_predictor import load_anchors

#: Career games at or under which an unanchored team is a scrimmage
#: opponent rather than a program. See the module docstring.
EXHIBITION_MAX_CAREER_GAMES = 5


def exhibition_teams(
    seasons: Iterable[Season],
    anchored: Iterable[str],
    namer: TeamNamer,
    max_career_games: int = EXHIBITION_MAX_CAREER_GAMES,
) -> frozenset[str]:
    """The teams whose games are exhibitions, by canonical name.

    Counted after the namer has had its say, so a program the registry knows
    under two spellings is one team with one count -- the same names the
    replay rates under.
    """
    career: Counter[str] = Counter()
    for season in seasons:
        for week in season.weeks:
            for game in week.games:
                game = namer.apply(game)
                career[game.home] += 1
                career[game.away] += 1
    filed = set(anchored)
    return frozenset(
        team
        for team, games in career.items()
        if games <= max_career_games and team not in filed
    )


def without_exhibitions(
    seasons: Sequence[Season], league: str, namer: TeamNamer | None = None
) -> list[Season]:
    """`seasons` with every exhibition game removed, weeks and all.

    Weeks are kept even when emptied, so week numbers and the season's
    calendar are what they were -- a week that held only a scrimmage still
    happened, and the deviations still age through it. Anchors are the
    league's own file, the same one the predictors read, and `namer`
    defaults to the league's registry the way `generate_predictions` does.
    """
    namer = namer if namer is not None else TeamNamer.for_league(league)
    junk = exhibition_teams(seasons, load_anchors(league), namer)
    if not junk:
        return list(seasons)

    def keep(game) -> bool:
        game = namer.apply(game)
        return game.home not in junk and game.away not in junk

    return [
        season._replace(
            weeks=[
                week._replace(games=[game for game in week.games if keep(game)])
                for week in season.weeks
            ]
        )
        for season in seasons
    ]


def exhibition_count(before: Sequence[Season], after: Sequence[Season]) -> int:
    """How many games `without_exhibitions` removed, for a run's log line."""
    return sum(len(w.games) for s in before for w in s.weeks) - sum(
        len(w.games) for s in after for w in s.weeks
    )

"""Team identity: which names are the same team.

The source feeds rename teams. ESPN has called the same D-III program
"North Central Cardinals" and "North Central College Cardinals", and the same
school "UW-Oshkosh Titans" and "Wisconsin-Oshkosh Titans". To a rating system
those are two teams: the old name keeps whatever rating it had when the feed
stopped using it -- forever, since nothing ever plays it again -- and the new
name starts over at 1500. Both halves are wrong, and the abandoned one is
worse, because a frozen rating never regresses and still sorts into the
rankings.

`call_it_what_you_want` owns the question of who a team is, so cassandra
doesn't answer it twice. It keys a team by its ESPN id -- the part that
survives a rename -- and records each name as an observation of what a source
called that team in a season, in a league. This module is the adapter: it
turns a registry into a rename that can be applied to a `Game`, and decides
what to do about the names a registry can't place.

Two things that look like details and aren't:

*Namespace, not league.* ESPN ids are unique per organization: one school
keeps its id across football and both basketballs, while the NFL numbers from
scratch and collides with the NCAA outright (id 2 is both the Buffalo Bills
and the Auburn Tigers). So all three college leagues share one registry, and
`nfl` deliberately has none rather than borrowing the wrong one.

*League, when asking for a name back.* The same team in the same season is
"UNLV Rebels" in football and "UNLV Lady Rebels" in women's basketball, so
every lookup here says which league is asking.

That policy is the other reason this isn't a one-liner. A replay covers 74k
games and a quarter century, and a name the registry can't place has to keep
playing rather than stop the run -- so unplaceable names pass through under
their own spelling, and get counted so the run can say how much it didn't
recognize.

Names are canonicalized here, not in the seasons in s3, so a bad correction is
undone by editing one file and re-running.
"""

from collections import Counter
from typing import Self

from call_it_what_you_want import (
    NCAA,
    NCAAFB,
    NCAAMBB,
    NCAAWBB,
    AmbiguousNameError,
    AmbiguousTeamError,
    NoNamesError,
    Teams,
    UnknownTeamError,
    default_teams,
)
from endgame.types import Game

# cassandra's league names are its own; these are what the registry calls the
# same thing. A league that isn't here gets no renaming at all.
_LEAGUES = {"ncaafb": NCAAFB, "mens": NCAAMBB, "womens": NCAAWBB}

# Which ESPN id namespace each league's teams are numbered in. `nfl` is
# absent on purpose: it's a different namespace, and the college one would
# resolve its team names to entirely different schools.
_NAMESPACES = {"ncaafb": NCAA, "mens": NCAA, "womens": NCAA}


def namespace_for(league: str) -> str | None:
    """The ESPN id namespace a league's teams are numbered in, if known."""
    return _NAMESPACES.get(league)


def registry_league(league: str) -> str | None:
    """What the registry calls one of cassandra's leagues.

    They agree on `ncaafb` and nowhere else -- cassandra's `mens` is the
    registry's `ncaambb` -- so anything writing a name observation has to
    translate rather than pass cassandra's name through. Recording a name
    under `mens` would file it in a league nothing ever asks about, and the
    name would then be invisible to every lookup.
    """
    return _LEAGUES.get(league)


class TeamNamer:
    """Renames a game's teams to one name per team, in one league's terms.

    Built from a `Teams` registry, so what counts as the same team is that
    package's answer and not a second one kept here.
    """

    def __init__(self, teams: Teams | None, league: str | None = None) -> None:
        self._teams = teams
        # What the *registry* calls this league, already translated, so the
        # hot path isn't looking it up per game.
        self._league = _LEAGUES.get(league) if league else None
        # Tallied rather than logged per game: a league with a gap in its
        # registry would otherwise print the same warning a few thousand
        # times, and the number is what actually matters.
        self.unknown: Counter[str] = Counter()
        self.ambiguous: Counter[str] = Counter()

    def __contains__(self, name: str) -> bool:
        """Whether any team answers to this name.

        Weaker than `resolves`, and the difference is the whole point: a
        name two teams share is *in* the registry and still can't be placed
        on one of them. Reporting those separately is what tells "nobody has
        heard of this school" apart from "two schools are called this".
        """
        return self._teams is not None and name in self._teams

    @classmethod
    def for_league(cls, league: str) -> Self:
        """The registry for a league's namespace.

        A league with no namespace -- `nfl`, or anything cassandra picks up
        later -- is not an error: it means every name passes through, which
        is what a league with stable names wants anyway.

        `default_teams` already layers whatever `sync_teams.py` recorded
        locally over what shipped with the package, so a rename found today
        is in effect today rather than waiting on a release. cassandra keeps
        no copy of its own; the staging file is the package's
        (`ciwyw where`), and `ciwyw show --output` is how those rows get
        upstreamed.
        """
        namespace = namespace_for(league)
        if namespace is None:
            return cls(None, league)
        try:
            return cls(default_teams(namespace), league)
        except ValueError:
            return cls(None, league)

    @classmethod
    def empty(cls) -> Self:
        """A namer that renames nothing. For tests, and for leagues with no data."""
        return cls(None)

    def canonical(self, name: str) -> str:
        """The one name for whatever team `name` refers to, in this league.

        `current_name` rather than the ESPN id: the ratings this feeds are
        published in a release and rendered on a front page, and "2426" is
        not a team as far as a reader is concerned. The id is still what
        does the matching -- it's just not what comes back.
        """
        if self._teams is None:
            return name
        try:
            return self._teams.current_name(name, league=self._league)
        except UnknownTeamError:
            # A team the registry has never seen. It keeps its own name, so
            # it rates normally and only misses out on being merged with its
            # other spellings.
            self.unknown[name] += 1
            return name
        except (AmbiguousTeamError, AmbiguousNameError, NoNamesError):
            # Either two teams answer to this name, or this team has no
            # name on record for this league, or it had two in one season.
            # None of those is something to guess at: renaming on a guess
            # merges two programs, which is far worse than leaving it alone.
            self.ambiguous[name] += 1
            return name

    def espn_id(self, name: str) -> str | None:
        """The team id `name` belongs to, or None if that isn't answerable.

        None rather than an exception for both failure modes, because every
        caller here is surveying a whole league's worth of names and wants a
        tally rather than a traceback. Unlike `canonical`, this doesn't count
        the miss -- the survey is the point, not a side effect of it.
        """
        if self._teams is None:
            return None
        try:
            return self._teams.espn_id(name)
        except (UnknownTeamError, AmbiguousTeamError):
            return None

    def resolves(self, name: str) -> bool:
        """Whether the registry can place this name on exactly one team."""
        return self.espn_id(name) is not None

    def same_team(self, left: str, right: str) -> bool:
        """Whether the registry already says these are one team.

        False when either name is unplaceable: two names it can't resolve
        may well be the same team, and saying "already handled" would hide
        exactly the pair a proposal run exists to find.
        """
        left_id = self.espn_id(left)
        return left_id is not None and left_id == self.espn_id(right)

    def apply(self, game: Game) -> Game:
        """The same game under canonical names.

        Returns the game itself when neither name changes, which is the
        common case once a registry covers a league.
        """
        home = self.canonical(game.home)
        away = self.canonical(game.away)
        if home == game.home and away == game.away:
            return game
        if home == away:
            # Both teams resolving to one team means the registry has two
            # programs under one id. Renaming would turn a real game into a
            # team playing itself, which the rating update would happily
            # process into nonsense.
            self.ambiguous[home] += 1
            return game
        return game._replace(home=home, away=away)

    def report(self) -> str:
        """One line on what didn't resolve, for the end of a run."""
        if self._teams is None:
            return "no team registry; names used as-is"
        return (
            f"{len(self._teams)} teams in registry; "
            f"{len(self.unknown)} unknown name(s), "
            f"{len(self.ambiguous)} ambiguous"
        )

    def __bool__(self) -> bool:
        return self._teams is not None

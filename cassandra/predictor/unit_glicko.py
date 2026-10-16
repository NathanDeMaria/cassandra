"""The margin model, with an offense and a defense per team rated on EPA per play.

`MarginGlickoPredictor` rates a team by its margins. This keeps that rating --
the *parent*, stepped exactly as the parent steps it -- and grows two *units*
under each team: an offense and a defense, each a Gaussian (mean, deviation)
pair in EPA per play, filtered on each offense's per-play average against the
defense it faced.

Why not `CompoundGlickoPredictor`
---------------------------------

That model has this shape and the wrong observation. It squashes a per-play
average through a logistic and hands it to `glicko_step`, which at its fitted
scales saturates about two contests in three; it rates on the garbage-time-
weighted reading, which split-half halves in reliability; and it centers on a
running mean over every season, which never caught the 2014 step in the
index. What an offline check says the units *should* be is simpler than any
of that: a per-season ridge,

    epa_off = center + edge + O[offense] - D[defense] + noise,  var ~ 1 / snaps

weighted by snaps, on the flat reading. Its top units on ncaafb are the ones
everyone would name (2019 LSU and Ohio State on offense, Clemson and Georgia
on defense; 2023 Michigan's defense; 2025 Indiana both ways), and it beats
raw EPA per play on the other half of a season by 0.05 R^2. The Kalman
filter below is that regression run forward one week at a time, and the
smoother (`passes`) re-walks the season toward it.

The contest
-----------

A game is two contests: the home offense against the away defense, and the
other way round. For one of them, with the offense at `O` (deviation `p`)
and the defense at `D` (`q`), both offsets in EPA per play:

    mu = center + (edge + a (anchor_off - anchor_def)) * c / (2 S) + O - D
    R  = play_sd^2 / snaps                       the reading's noise
    v  = epa - mu
    O += p^2 / (p^2 + q^2 + R) * v               p^2 *= 1 - p^2 / (p^2 + q^2 + R)
    D -= q^2 / (p^2 + q^2 + R) * v               q^2 *= 1 - q^2 / (p^2 + q^2 + R)

which is the joint Kalman step on the two states, each side's marginal, with
the other side's deviation in the noise as the parent folds in an opponent's.
`S` is `points_per_epa` -- margin points per EPA per play of differential
between two offenses over a game, 75 on ncaafb -- and `c` is the parent's
`points_per_rating`, so `c / (2 S)` turns a rating gap into the per-play gap
each contest carries half of. The anchor term is what makes the anchors
priors for the units: an FBS offense against a D-II defense is expected to
move the ball, and a unit at offset 0 is a unit that did what its tier does.
`a` is `anchor_epa_scale`, 1 unless set; see "Across conferences". The edge
is the parent's own -- home advantage and matchup terms -- split between the
contests the same way.

The center is the league's average offense, per season: this season's
running mean, pulled toward last season's by `CENTER_PRIOR_SIDES` sides of
it, so a season opens where the last one closed and moves when the league
does. The step at 2014, where the feed became legible, is a season-to-season
move of 0.1 per play, and a mean over every season never catches it.

Offense and defense on their own clocks
---------------------------------------

Each side has its own starting deviation (the cap it grows back to), weekly
and seasonal increase, and offseason regression. For a win probability only
`O + D` counts -- a team's margin against another is `S` times the gap in
the two sums -- so if the two sides ran on one clock this would be a net EPA
rating with extra bookkeeping. The split is worth having only because the
two sides are known and forgotten at different rates, and measured, they
are: see below. It isn't the way round it is usually said. The offense is
the side that moves *during* a season (a quarterback is one injury from a
different unit) and the side that forgets more over the summer; a defense
is closer to fixed for the year and carries more of itself into the next.

How the units speak
-------------------

As `CompoundGlickoPredictor`'s do: by precision, at prediction, on the
parent's rating scale. A team's unit rating is

    anchor + (S / c) * (O + D)

with variance `(S / c)^2 (p^2 + q^2)`, and the rating predicted from is

    (parent / d^2 + w * earned * unit) / (1 / d^2 + w * earned)

where `earned` is the unit's precision net of what it started with -- 0 for
a team the index never had a play for, which stays at its record -- and `w`
is `unit_weight`. At `unit_weight` 0 this is `MarginGlickoPredictor` exactly,
game by game; the units never feed back into the parent's step and the
parent never sees an EPA number.

What changed over the summer
----------------------------

Three knobs read `cassandra.offseason`'s facts, each a shift in points
applied to the team's rating and its offense together, so the blend moves
by the same amount. All three are 0 by default, which is the model without
them.

- `coach_left_shift`, at the rollover, for a team whose head coach left for
  a better job. That is known before the season, so it is applied then.
- `new_qb_shift` plus `qb_quality_shift` per 0.1 EPA per attempt of
  quarterback quality change, for a team whose week-one starter is new.
  Applied after the team's first game with play-by-play, not at the
  rollover: who starts is only in the plays once he has, and a model that
  read it any earlier would be scoring itself with a depth chart it doesn't
  have. Not after an opener without plays, either -- the starter on file is
  then the next game's, and shifting before that game would read its own
  starter.

A shift applied mid-season lands on the preseason ratings the smoother
re-walks from as well, or the next pass would undo it. Replayed on ncaafb
2002-2025 on the fitted units model, against no shifts, on 2015+ FBS
games (and their first four games):

    shifts                                           FBS         games 1-4
    quarterback, after game 1, measured sizes     -0.00043      -0.00063
    (the rows above waited for game 1 even without plays; the two below
    wait for a game with plays, and read one game less ahead)
    quarterback, after game 1, 1.5x               -0.00057      -0.00077
      the same, before the season (a depth chart) -0.00074      -0.00116
    coach left for a job, -2.5                    -0.00013      -0.00023
    quarterback 1.5x after game 1 + coach -2.5    -0.00066      -0.00089
      and the offense's starting sd at 0.10       -0.00085      -0.00153

The last row is the clocks moving with the shifts: once the summer's change
at quarterback is read directly, an offense no longer needs the blanket
uncertainty that stood in for it (0.14 was the best starting sd without
the shifts). `glicko_margin_units_offseason` searches the shifts with the
offense's starting sd and offseason regression, the rest pinned.

What the roster says
--------------------

Four knobs read `cassandra.roster`: each FBS team's 247 talent composite and
its returning offensive and defensive production, all in standard
deviations from that season's FBS mean. All four are 0 by default.

- `offense_returning_shift` and `defense_returning_shift`, points per sd at
  the rollover, moving the team's rating by this season's level every
  summer. Returning production says how this team differs from last
  season's -- whose games the rating has since priced -- not a standing
  level, so last summer's shift isn't taken back.
- `talent_shift`, points per sd of talent at the rollover, applied the way
  the shifts above are, rating and offense -- but only the *change* since
  the team's last rated season, so the summers' shifts add up to
  `talent_shift` times this season's level and never more. Talent is a
  standing level (0.96-0.99 year to year); shifted by the level every
  summer, what one summer's shift left behind carried into the next, and a
  shift of 1 point per 100 talent moved the first games' margins by 0.8 per
  100 in 2015 and by 1.4 by 2021.
- `talent_edge`, points of margin per sd of the two teams' talent gap, added
  to every game between two rated teams as a matchup term -- home field's
  kind of number rather than a rating's. It is in the update as well as the
  prediction, so the ratings learn around it, and it never carries into the
  next season.

The returning shifts move the rating and not the units. The units are
learned from plays, and the defense hardly regresses between seasons
(`defense_season_regression` ~0.06), so a summer's shift parked in a defense
unit would still be there the next summer and the one after; offense and
defense through their own units gained 13.92 where the rating alone gains
14.69, at 2 points each with the talent edge at 2.

Returning production was a matchup term first, like the talent edge. It
gained as much on FBS v FBS games but cost the FBS games against lower
tiers, whose teams have no values: a rating learned around the term in its
FBS games and took that into games the term didn't reach. A shift is in the
rating, so it reaches every game -- and a release carries it.

Widening a team's offseason deviation by its turnover, as well as or
instead of shifting it, costs: scaling the season increase by
`exp(-0.25 z)` gave back 14e-4 on FBS v FBS games, because the deviation
also sets how sure every prediction for the team is, not only how fast the
rating moves.

Replayed on the fitted `glicko_margin_units_offseason`, brier change against
none of it, in units of 1e-4, on FBS v FBS games 2015-2025 (by a side's game
of the season), FBS against lower tiers 2015-2025, and 2026's 265 FBS games.
Defense starts in 2017, so its 2015-2016 seasons are the model without it:

    points per sd                         2015-25  games 1-4  games 5+  lower tier   2026
    talent edge 2                           -1.16      -0.75     -1.34       -0.69  -9.59
    talent change-only shift 2              -0.86      -0.07     -1.20       +0.01  +1.50
    talent level shift 1, every summer      -0.18      -0.40     -0.08       +1.77 -15.50
    talent edge 2, returning edges 1.5     -15.43     -17.85    -14.40       +4.68 -39.46
    talent edge 2, returning shifts 2      -14.69     -17.50    -13.50       -2.80 -41.44
    talent edge 2, returning shifts 2.5    -15.51     -16.40    -15.13       -2.38 -43.87

The last row gains in eleven seasons of twelve. The talent forms alone all
gain in the same seasons and lose in the same four -- 2020, 2021, 2023 and
2025 -- so which seasons gain is how much talent mattered that year, not
which form read it.

The talent edge is read here rather than as a `MatchupAdjustments` term
because talent is a fact about a season, and a `Matchup` doesn't carry one
-- a January bowl's date is the next year's -- where the predictor's clock
does. Like the offseason facts, the indexes stay out of `state_dict`. The
shifts are already in a release's ratings; the talent edge needs the file,
so a model rebuilt from a release prices it only if the league's file is on
disk.

Success rate
------------

The units can read a second number off the same snaps: the share of an
offense's dropbacks and runs that added expected points. `success_weight`
blends it in -- 0 is EPA per play, the model above; 1 is success rate
alone -- after `SUCCESS_SCALE` has put it on EPA's scale, so the blend is
between two numbers in the same units and every other knob means what it
did. The blended reading is what the filter steps on, what the center
averages, and is priced over EPA's snap count.

Why it helps is the reason it reads worse on the game it came from. A
game's EPA differential follows its margin closely (r 0.93 on the NFL,
2014-2026) because it is mostly the same few big plays the scoreboard is;
its success-rate differential follows it loosely (0.61) because a long
touchdown is one success in sixty. What a rating wants is the part that
comes back next week, and the big plays mostly don't.

Replayed on the NFL off the published `glicko_margin_units_offseason`, each
weight with the eight knobs in the reading's units (`points_per_epa`,
`play_sd`, both initial deviations, both weekly and both season increases)
refit by Nelder-Mead on 1980-2021, and the same refit given to EPA alone.
Brier against that EPA refit with the offense's weekly increase held at
0.02 -- searched free, EPA's refit stuck at 0, a worse optimum -- in units
of 1e-4, on all games of the years shown and on the 792 since 2014 with a
starting quarterback flagged out:

    success_weight                  2014-2026   held out, 2022-2026   QB out
    0.25                              +1.1             +0.7            +2.4
    0.5                               -3.8             -4.6            -4.7
    0.75                              -5.3             -7.7           -11.2
    1.0                               -4.7             -6.6           -12.5

0.75 against the published model: -8.5 on 2014-2026 (t -2.1 on the
per-game differences, better in 11 seasons of 13 and all five held out).
Success rate and the weekly forgetting each carry about half: the 0.75
blend with no weekly increase gains -3.2 on EPA with none, and the
increase is worth -2.5 to EPA and -4.5 to the blend. Every success-rate
fit chooses an offense that forgets about 0.02 a week -- an injured
quarterback's offense showing up in its results within a few games -- which
on EPA alone, noisier game to game, pays for itself only barely. The games
it buys most on are the ones the qb_out index flags, which is what the
forgetting is for.

`models/nfl/glicko_margin_units_success.json` searches the weight, those
eight and the coach knobs below on the offseason model's other pins, and
`qb_out_penalty` with them: an offense that forgets in a few weeks
already carries some of what the penalty was fit to say.

`success_reading` picks which success rate: `"unweighted"`, every dropback
and run once, or `"weighted"`, each by the competitiveness weight the
weighted EPA uses, so the fourth quarter of a rout counts for what it is
worth. Each has its own constants in `SUCCESS_SCALE`. On the NFL the
weighted rate is worse: at the 0.75 fit's knobs, -7.4 on 2014-2026 against
the published model where the unweighted gains -8.5, and -5.6 held out
against -9.5. On ncaafb the two are even, -0.2 (t -0.3) at the published
roster model's knobs and -0.2 with each refit.

ncaafb's whole-model refits don't carry over. Replayed through the
published `glicko_margin_units_roster`, with the eight knobs and
`unit_weight` refit on FBS v FBS brier through 2021: every success-rate
fit -- unweighted, weighted, or blended with the weighted EPA -- goes to
weight 1 and beats the same refit on EPA by 1.6 to 2.6 on FBS games
2014-2026 and loses by 2.5 to 3.1 on the held-out 2022-2026, no |t|
above 1 either way, while one season alone swings by 20. The loss is the
refit's clocks rather than its reading: at those knobs EPA itself is 5.6
worse held out.

`defense_success_weight` gives the defenses a blend of their own -- None,
the default, reads `success_weight` for both. A contest then carries two
readings, and each unit steps on its own innovation against the same
expectation. Success rate is an offense's number. On the NFL, at the
joint refit's knobs below, the offense's weight from 0 to 1 moves
2014-2026 by -12.5 and the defense's by 0.3 at most -- though the NFL's
fits hold the defense nearly still whatever it reads. On ncaafb, at the
EPA refit's knobs, the offense alone at 0.5 gains 1.4 on FBS games
2014-2026 (t -1.5, 9 seasons of 13), 1.4 held out and 2.5 after a
team's fourth game (t -2.7), and costs 1.0 in its first four; the
defense alone gains 0.1 and loses 1.7 held out. Refit with the two weights
free, the NFL's defense goes to 0 and ncaafb's to 1, which wins through
2021 and loses held out.

So `models/ncaafb/glicko_margin_units_success.json` searches the offense's
weight alone, with `points_per_epa`, `play_sd` and `unit_weight`, on the
roster model's pins, the defenses on EPA. The same four refit through 2021
from the published roster model: the offense goes to 0.59 and gains 0.9 on
FBS games 2014-2026 against EPA given the same refit (t -1.0), 1.0 held
out, 1.5 after a team's fourth game (t -1.9) and 3.4 in Group of Five
games (t -2.5). Both fits go to `unit_weight` 1, where the replay capped
them. Let past it, both run on to 5-6 with `play_sd` near 3, win through
2021 and lose held out (0.9 with success rate, 2.1 without, 2 seasons of 5
better), and the first search over every season (run 20261014-010003,
`unit_weight` up to 2) stopped at 1.92. The config caps it at 1; success
rate's own share held in every one of these, 1.0 to 2.2 held out.

Across conferences
------------------

ncaafb's anchors are one rating per conference, so a conference game has
no anchor gap in it and teaches a side only where it stands among its own.
A conference's level is its anchor, made a per-play gap by `points_per_epa`,
and what the few games across conferences say about it. `anchor_epa_scale`
sets that conversion apart from the one the offsets go back through: how
far a reading is expected to move across an anchor gap, as a share of what
`points_per_epa` makes of it. 1, the default, is the model above.

Whether a reading moves across conferences the way it moves within one is
the question a success rate raises: a talent gap may show more in the big
plays than in the share of downs won. Replayed on ncaafb 2014-2026, the
surprise in an FBS offense's reading against another conference's defense
-- what it read less what the units expected -- falls with the anchor gap
at a slope of -0.25 under the published model (t -15), whose units end
every season with the power conferences' sides 8 to 12 points under the
Group of Five's against their anchors; at +0.04 under the EPA refit, whose
`points_per_epa` is twice the published one's; and at -0.06 and -0.09
under the unweighted and weighted success-rate refits, which leave the
power conferences 1 to 8 points under. But no scale changes brier by as
much as 1e-4 on FBS games -- 0.75 on the published model, 0.85 to 0.95 on
the weighted success rate's fit, 1.05 on EPA's -- and every one moves the
slope away from 0 rather than toward it. So the slope isn't only the
conversion: the offsets also carry when in a season the games across
conferences come, and what the parent rating already knows of them.
Searchable, and 1 until a search moves it.

A new head coach
----------------

`coach_change_rd`, `coach_change_offense_sd` and `coach_change_defense_sd`
widen a team's record and each of its sides at the rollover when its head
coach has changed, for any reason -- a fired coach's successor is as new
as a poached one's, and what the deviation says is how much the team is
still to be learned, not which way. The deviation, not a shift: a coach
leaving for a job is a known direction (`coach_left_shift`), and a coach
changing is a known unknown. Each side by its own, because a new staff
usually means a new scheme on both sides, but not one known in advance.
A side can go past its initial deviation, which is a fitted clock rather
than a bound -- the NFL's defense sits on its 0.05 all season -- and stays
there until its games shrink it.

On the NFL it is the record's deviation that pays, and not by much.
Refit with the success-rate blend's eight knobs and its weight on
1980-2021, from the 0.75 fit: `coach_change_rd` 62 -- about 7.5 points of
margin -- both sides' about 0, and the weight to 0.85. Against the same
fit without them, in 1e-4 of brier:

                                  2006-2026   held out, 2022-2026   new coach   its games 1-4
    joint refit                     -1.0            -1.2              -3.0          -1.5
    coach knobs alone               -0.8            -1.1              -1.3          +6.3
    coach knobs on EPA alone        -1.1            -1.9              -2.2          +1.9

with no |t| above 1.4 on the per-game differences. Fit alone on the 0.75
knobs, the search took the offense's deviation to where a new coach's
first game rewrites it, which is what the early games pay for; refit
together, the sides settle at zero and the early games stop paying. It is
what an ncaafb prototype found in 2026-09, before these knobs existed: a
little from the record's deviation, and a side's widening costing games
1-4. They default to 0; the NFL's success-rate config searches them.

What a release carries
----------------------

`ratings` hands each side out on the team scale -- `anchor + 2 (S / c) O`,
so the mean of a team's two sides is its unit rating -- and `from_ratings`
takes them back. The center isn't carried, which only matters to a rebuilt
model that goes on updating; a prediction never reads it.

And each side's success rate, for a reader (`Rating.success`). What the
units read is a blend on EPA's scale, or EPA alone, so no side carries a
success rate a person would recognize. Beside them, every units model keeps
a second pair per team that reads the flat success rate only: the same
contest, the same anchors and edge (turned into a rate through
`points_per_epa`, the league's `SUCCESS_SCALE` and a slope the replay
measures), a center that is the league's average rate, and a binomial
reading, `p (1 - p) / downs`. Out of
it comes each offense's rate against an average defense on a neutral field,
and the rate each defense allows an average offense -- `center +` the
team's anchor against an average opponent's `+` its offset -- with the
offset's deviation as the uncertainty. An average opponent is the upper
tier's: on ncaafb the FBS (`_SuccessRates`).

Nothing predicts from it, so no model's numbers move, and it doesn't go
through the smoother, coach widening or quarterback shifts, which are about
what to predict. Its clocks are measured rather than searched
(`SUCCESS_RATE_CLOCKS`), and `from_ratings` drops it: a rebuilt model that
goes on updating starts it again, which is a display coming back, not a
prediction moving. A release written before it reads as before, and a
reader from before it ignores the field.

Replayed through the published models, every prediction is unchanged. On
ncaafb in 2026 the average FBS offense reads 42.8% and the average FBS
defense allows 42.1%, against a league center of 42.1%; the best offenses
sit near 50% (Georgia, Ohio State), the best defenses near 33% (Oklahoma).
The NFL's best offenses read 48-49%, its best defenses 44-45%, around a
center of 46.2%. On ncaafb every units model reads the same rates: the slope
absorbs `points_per_epa`, 0.38 at the roster model's 75 and 0.44 at the
success model's 87. The NFL's differ in the third place, its slope fit on
the home edge alone.

What it was measured to be worth
--------------------------------

Replayed on ncaafb 2002-2025 off the local season cache, on the fitted
`glicko_margin` parent (run 20260921; brier here 0.154604, since the replay
skips the opponent-prior warm-up), about forty hand probes. Brier against
`unit_weight` 0, which is the parent exactly:

    units                                all      EPA games  2014+ FBS  2014+ FCS  FBS-FCS
    first draft (weight 0.5, ridge
      clocks, play_sd 1.3)             -0.000049  -0.000343  -0.000321  +0.000403  -0.003946
    one clock for both sides, best     -0.000102  -0.000500  -0.000632  +0.000428  -0.003683
    split clocks (the defaults)        -0.000223  -0.000900  -0.000846  -0.000506  -0.003351

The split is worth as much again as the units on one clock, and it is what
turned FCS-vs-FCS from a loss into a gain. The probes that said which way
round, off the one-clock row: a weekly increase of 0.02 on the offense and
none on the defense gains 0.00037 on 2014+ FBS, the reverse costs 0.00025;
an offseason regression of 0.8 on the offense and 0.4 on the defense beats
the reverse; a season increase of 0.10 offense and 0.03 defense beats the
reverse by 0.0006 on FBS.

For scale: the best `CompoundGlickoPredictor` variant found on the same
cache gained 0.00004 on FBS-vs-FBS and the shipped one lost there. The
FBS-vs-FCS column is where the anchors' ladder was already known to be
off (the margin is over-dispersed across the tier gap), and a unit that
knows how an FCS offense moves against FBS defenses fixes a lot of it.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, Self

from endgame.types import Game

from ..offseason import OffseasonFacts
from ..roster import DEFENSE, OFFENSE, TALENT, RosterIndex
from .adjustments import (
    DEFAULT_QB_OUT_PENALTY,
    DEFAULT_TRAVEL_ADVANTAGE,
    MatchupSources,
)
from .base_predictor import Anchor, validated_regression
from .blend import validated_scale
from .distribution import MarginDistribution, NormalMargin
from .epa import EpaIndex
from .glicko import _Rating
from .margin_glicko import (
    DEFAULT_OBS_SD,
    DEFAULT_POINTS_PER_RATING,
    MarginGlickoPredictor,
)
from .opponent_prior import OpponentPriorManager
from .rest import DEFAULT_REST_ADVANTAGE
from .types import (
    GameEpa,
    Matchup,
    Prediction,
    Rate,
    Rating,
    SuccessRates,
    Unit,
    Units,
)

#: Margin points per EPA per play of differential between the two offenses,
#: over a game. The slope of the margin on `home - away` per-play EPA over
#: ncaafb 2014-2025 is 73.5 (75.0 FBS-vs-FBS), which is the median 78 snaps
#: a side times a clip that understates blowouts a little.
DEFAULT_POINTS_PER_EPA = 75.0

#: Noise of one snap's clipped EPA about what the matchup predicted, so a
#: side's per-play average over `n` snaps reads with variance `play_sd^2 / n`.
#: The ridge's residuals put it between 1.1 (in hindsight, which understates
#: it) and 1.6; replayed, 1.0 beat 1.3 and 0.8 was worse again.
DEFAULT_PLAY_SD = 1.0

#: Where a unit's deviation starts, in EPA per play, and the cap it grows
#: back to. The within-division spread of adjusted units is 0.08 on FBS;
#: 0.14 leaves room for the conference the anchor doesn't see, and replayed
#: beat 0.10 and 0.18.
DEFAULT_OFFENSE_INITIAL_SD = 0.14
DEFAULT_DEFENSE_INITIAL_SD = 0.14

#: How much a unit's deviation widens each week. The offense moves in
#: season and the defense doesn't; see the module docstring.
DEFAULT_OFFENSE_WEEKLY_SD_INCREASE = 0.05
DEFAULT_DEFENSE_WEEKLY_SD_INCREASE = 0.0

#: How much a unit's deviation widens each offseason. The ridge's
#: year-to-year change is 0.069 offense and 0.063 defense; replayed, the
#: offense wants more room than that and the defense much less.
DEFAULT_OFFENSE_SEASON_SD_INCREASE = 0.10
DEFAULT_DEFENSE_SEASON_SD_INCREASE = 0.03

#: How far each offseason pulls a unit back to its anchor. The ridge says
#: both sides keep 0.55 of themselves; replayed, the offense keeps 0.2 and
#: the defense 0.8.
DEFAULT_OFFENSE_SEASON_REGRESSION = 0.8
DEFAULT_DEFENSE_SEASON_REGRESSION = 0.2

#: How far to trust a unit's deviation against the parent's at prediction.
#: The units' deviations are a Gaussian filter's, honest about EPA as a
#: reading of EPA; what they aren't honest about is how much of EPA is the
#: margin. 0.2 to 0.25 replayed best, 0.35 and 0.5 were worse.
DEFAULT_UNIT_WEIGHT = 0.2

#: The departure category that moves a team's rating over the summer; see
#: `cassandra.offseason` for why the others don't.
LEFT_FOR_JOB = "left_for_job"

#: How many sides of last season's average offense this season's center
#: starts with. A few weeks of games outweigh it.
CENTER_PRIOR_SIDES = 200

#: The two success rates a game carries: every dropback and run once, or
#: each weighted by how competitive the game still was. See "Success rate".
UNWEIGHTED = "unweighted"
WEIGHTED = "weighted"

#: How a success rate reads on EPA's scale, per league and reading: `scale *
#: rate + level`, so `success_weight` blends two numbers in the same units and
#: every other knob keeps its meaning. The scale is the ratio of the two
#: same-game margin slopes -- points of margin per unit of success-rate gap,
#: over points per EPA per play of gap -- and the level puts the league's
#: average success rate where its average EPA per play is. Only the scale
#: matters to a blended reading, whose center absorbs a constant; the level
#: is there so a game without a success rate, read at its EPA alone, lands
#: on the same footing as the rest.
#: Measured on each league's index from 2014, against 65.1 points per EPA
#: per play on the NFL's 3,369 games and 73.5 on ncaafb's 17,087: a unit of
#: success rate is 88.7 and 85.4 points on the NFL, unweighted and weighted,
#: and 132.1 and 109.8 on ncaafb -- where the weighting takes the most out,
#: since its gaps were widest in the games already decided.
SUCCESS_SCALE: Mapping[str, Mapping[str, tuple[float, float]]] = {
    "nfl": {UNWEIGHTED: (1.362, -0.645), WEIGHTED: (1.311, -0.621)},
    "ncaafb": {UNWEIGHTED: (1.798, -0.751), WEIGHTED: (1.494, -0.624)},
}


class RateClock(NamedTuple):
    """One side's clock for the displayed success rate; see "What a release carries"."""

    #: The spread of true rates: where a side starts, and the cap it widens to.
    initial_sd: float
    #: The share of a side's offset it keeps over the summer, its deviation
    #: refilling toward `initial_sd` as the rest is forgotten (AR(1)).
    carry: float
    #: How much a side's deviation widens each week.
    weekly_sd: float


#: The displayed success rate's clocks, offense then defense, per league.
#: The spreads and carries are `reasoning/success_rate/combine/
#: persistence.py`'s: the true spread of each side's opponent-adjusted
#: efficiency, over EPA per play on the success rate (1.794 ncaafb, 1.479
#: NFL), and its season-to-season correlation over its reliability. The
#: weekly widening is the NFL's measured drift, 0.0037 logits a week, about
#: 0.001 of a rate; ncaafb's isn't measured and takes the same.
SUCCESS_RATE_CLOCKS: Mapping[str, tuple[RateClock, RateClock]] = {
    "ncaafb": (RateClock(0.050, 0.85, 0.001), RateClock(0.044, 0.86, 0.001)),
    "nfl": (RateClock(0.027, 0.58, 0.001), RateClock(0.019, 0.59, 0.001)),
}

#: A rate the binomial reading is priced at, kept off the ends.
_RATE_FLOOR = 0.05

#: How much evidence the displayed rate's gap slope starts with, at 1, in
#: squared rate of expected gap: about forty games across a 0.05 gap.
GAP_SLOPE_PRIOR = 0.1


class _Sides(NamedTuple):
    """One team's offense and defense, as offsets from its anchor, in EPA per play."""

    offense: _Rating
    defense: _Rating


class _Contest(NamedTuple):
    """One offense against one defense, as the forward pass saw it.

    `base` is what the contest was expected to read before either unit's
    offset: the center, the anchors and the edge as they stood that day. Kept
    rather than re-derived for the reason `_Played` keeps the home edge: the
    center and the rest ledger have moved on by the time the smoother runs.
    """

    offense: str
    defense: str
    epa: float
    noise: float
    base: float
    # What the defense's step reads, where it differs from the offense's --
    # see `defense_success_weight`. None is the same reading.
    defense_epa: float | None = None


class UnitMarginGlickoPredictor(MarginGlickoPredictor):
    """The margin model, with offense and defense units rated on EPA per play.

    The parent's every method does what it did. What this adds is `_sides`,
    stepped from the EPA index after the parent has stepped on the margin,
    and read by `predict_distribution` as a second rating to combine with
    the first. See the module docstring for the step and the combination.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        home_advantage_slope: float = 0.0,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        obs_sd: float = DEFAULT_OBS_SD,
        nu: float | None = None,
        points_per_rating: float = DEFAULT_POINTS_PER_RATING,
        prediction_scale: float | None = None,
        mismatch_scale: float | None = None,
        # Both the parent's ratings and the units are re-walked this many
        # times a week; see `GlickoPredictor.passes`.
        passes: int = 1,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        unit_weight: float = DEFAULT_UNIT_WEIGHT,
        points_per_epa: float = DEFAULT_POINTS_PER_EPA,
        play_sd: float = DEFAULT_PLAY_SD,
        offense_initial_sd: float = DEFAULT_OFFENSE_INITIAL_SD,
        defense_initial_sd: float = DEFAULT_DEFENSE_INITIAL_SD,
        offense_weekly_sd_increase: float = DEFAULT_OFFENSE_WEEKLY_SD_INCREASE,
        defense_weekly_sd_increase: float = DEFAULT_DEFENSE_WEEKLY_SD_INCREASE,
        offense_season_sd_increase: float = DEFAULT_OFFENSE_SEASON_SD_INCREASE,
        defense_season_sd_increase: float = DEFAULT_DEFENSE_SEASON_SD_INCREASE,
        offense_season_regression: float = DEFAULT_OFFENSE_SEASON_REGRESSION,
        defense_season_regression: float = DEFAULT_DEFENSE_SEASON_REGRESSION,
        # Offseason shifts, in points; see "What changed over the summer".
        coach_left_shift: float = 0.0,
        new_qb_shift: float = 0.0,
        qb_quality_shift: float = 0.0,
        talent_shift: float = 0.0,
        talent_edge: float = 0.0,
        offense_returning_shift: float = 0.0,
        defense_returning_shift: float = 0.0,
        # The reading the units filter on: 0 is EPA per play, 1 success
        # rate, and which success rate; see "Success rate".
        success_weight: float = 0.0,
        success_reading: str = UNWEIGHTED,
        # The defenses' own blend; None reads `success_weight` for both.
        defense_success_weight: float | None = None,
        # How far a reading is expected to move across an anchor gap, as a
        # share of what `points_per_epa` converts it to; see "Across
        # conferences".
        anchor_epa_scale: float = 1.0,
        # Offseason deviation for a team whose head coach changed; see
        # "A new head coach". Rating points for the record, EPA per play for
        # the units, all 0 by default.
        coach_change_rd: float = 0.0,
        coach_change_offense_sd: float = 0.0,
        coach_change_defense_sd: float = 0.0,
        opponent_prior_manager: OpponentPriorManager | None = None,
        sources: MatchupSources | None = None,
        ratings: dict[str, _Rating] | None = None,
        sides: dict[str, _Sides] | None = None,
        # The center's state: last season's average offense (None before
        # any season with plays), and this season's sum and count of sides.
        # Carried so a loaded model keeps centering where the replay left it.
        epa_center_state: Sequence[float | None] = (None, 0.0, 0),
        # The displayed success rates' sides and center, carried like the
        # units'; see "What a release carries".
        success_rates_state: Mapping[str, Any] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        unanchored_seen: Sequence[float] = (0.0, 0.0, 0),
        game_epa: EpaIndex | None = None,
        offseason: OffseasonFacts | None = None,
        talent: RosterIndex | None = None,
        offense_returning: RosterIndex | None = None,
        defense_returning: RosterIndex | None = None,
    ) -> None:
        super().__init__(
            league,
            home_advantage=home_advantage,
            home_advantage_slope=home_advantage_slope,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            obs_sd=obs_sd,
            nu=nu,
            points_per_rating=points_per_rating,
            prediction_scale=prediction_scale,
            mismatch_scale=mismatch_scale,
            passes=passes,
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            sources=sources,
            ratings=ratings,
            anchors=anchors,
            unanchored_seen=unanchored_seen,
        )
        if unit_weight < 0:
            # 0 is the off switch. Below it a unit's evidence counts against
            # the parent's, which is a model that learns backwards.
            raise ValueError(f"unit_weight must be non-negative, got {unit_weight}")
        self._unit_weight = unit_weight
        self._points_per_epa = validated_scale("points_per_epa", points_per_epa)
        self._play_sd = validated_scale("play_sd", play_sd)
        self._initial_sd = _Pair(
            validated_scale("offense_initial_sd", offense_initial_sd),
            validated_scale("defense_initial_sd", defense_initial_sd),
        )
        self._weekly_sd_increase = _Pair(
            _non_negative("offense_weekly_sd_increase", offense_weekly_sd_increase),
            _non_negative("defense_weekly_sd_increase", defense_weekly_sd_increase),
        )
        self._season_sd_increase = _Pair(
            _non_negative("offense_season_sd_increase", offense_season_sd_increase),
            _non_negative("defense_season_sd_increase", defense_season_sd_increase),
        )
        self._unit_regression = _Pair(
            validated_regression(offense_season_regression),
            validated_regression(defense_season_regression),
        )
        self._sides: dict[str, _Sides] = sides if sides is not None else {}
        previous, total, count = epa_center_state
        self._center_previous = None if previous is None else float(previous)
        self._center_sum = float(total or 0.0)
        self._center_count = int(count or 0)
        self._game_epa = game_epa or EpaIndex.for_league(league)
        self._success_rates = _SuccessRates(league, success_rates_state)
        # The smoother's ledger for the units, as the parent keeps one for
        # the ratings: where the season's sides started, and its contests.
        self._sides_preseason = dict(self._sides)
        self._contest_weeks: list[list[_Contest]] = []
        self._contests_this_week: list[_Contest] = []
        self._coach_left_shift = coach_left_shift
        self._new_qb_shift = new_qb_shift
        self._qb_quality_shift = qb_quality_shift
        # Defaulted like `game_epa`: every caller wants the league's facts,
        # and a test that wants none passes `OffseasonFacts()` -- checked
        # against None, since an empty one is falsy. Not carried in
        # `state_dict`, like the index.
        self._offseason = (
            offseason if offseason is not None else OffseasonFacts.for_league(league)
        )
        # Roster talent; see "Roster talent". Non-negative, as more talent
        # being worse is not a hypothesis worth a search's time.
        self._talent_shift = _non_negative("talent_shift", talent_shift)
        self._talent_edge = _non_negative("talent_edge", talent_edge)
        self._offense_returning_shift = _non_negative(
            "offense_returning_shift", offense_returning_shift
        )
        self._defense_returning_shift = _non_negative(
            "defense_returning_shift", defense_returning_shift
        )
        self._talent = (
            talent if talent is not None else RosterIndex.for_league(league, TALENT)
        )
        self._offense_returning = (
            offense_returning
            if offense_returning is not None
            else RosterIndex.for_league(league, OFFENSE)
        )
        self._defense_returning = (
            defense_returning
            if defense_returning is not None
            else RosterIndex.for_league(league, DEFENSE)
        )
        # Teams whose quarterback shift waits on their first game this
        # season. In-season state, like the smoother's ledger.
        self._qb_pending: set[str] = set()
        if defense_success_weight is None:
            defense_success_weight = success_weight
        for name, weight in (
            ("success_weight", success_weight),
            ("defense_success_weight", defense_success_weight),
        ):
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {weight}")
        if (success_weight or defense_success_weight) and league not in SUCCESS_SCALE:
            # A blend needs the league's scale, and a guessed one is a
            # weight nobody chose.
            raise ValueError(f"no success-rate scale for {league!r}")
        if success_reading not in (UNWEIGHTED, WEIGHTED):
            raise ValueError(
                f"success_reading must be {UNWEIGHTED!r} or {WEIGHTED!r}, got {success_reading!r}"
            )
        self._success_weight = success_weight
        self._defense_success_weight = defense_success_weight
        self._success_reading = success_reading
        self._anchor_epa_scale = _non_negative("anchor_epa_scale", anchor_epa_scale)
        self._coach_change_rd = _non_negative("coach_change_rd", coach_change_rd)
        self._coach_change_sd = _Pair(
            _non_negative("coach_change_offense_sd", coach_change_offense_sd),
            _non_negative("coach_change_defense_sd", coach_change_defense_sd),
        )

    @property
    def unit_weight(self) -> float:
        """How far a unit's deviation is trusted against the parent's."""
        return self._unit_weight

    @property
    def points_per_epa(self) -> float:
        """Margin points per EPA per play of differential, over a game."""
        return self._points_per_epa

    @property
    def epa_center(self) -> float:
        """The league's average offense, in EPA per play, as this season has it.

        This season's running mean, with last season's as a prior worth
        `CENTER_PRIOR_SIDES` sides. 0 before anything has been seen.
        """
        if self._center_previous is None:
            if not self._center_count:
                return 0.0
            return self._center_sum / self._center_count
        return (self._center_previous * CENTER_PRIOR_SIDES + self._center_sum) / (
            CENTER_PRIOR_SIDES + self._center_count
        )

    @property
    def _rating_per_epa(self) -> float:
        """Rating points of team gap per EPA per play of `O + D`."""
        return self._points_per_epa / self.points_per_rating

    def _offsets(self, team: str) -> _Sides:
        return self._sides.get(
            team,
            _Sides(
                _Rating(0.0, self._initial_sd.offense),
                _Rating(0.0, self._initial_sd.defense),
            ),
        )

    def get_sides(self, team: str) -> _Sides:
        """This team's offense and defense offsets from its anchor, in EPA per play."""
        return self._offsets(team)

    def unit_rating(self, team: str) -> float:
        """The team as its units see it, on the parent's rating scale."""
        sides = self._offsets(team)
        return self.anchor(team) + self._rating_per_epa * (
            sides.offense.rating + sides.defense.rating
        )

    def unit_information(self, team: str) -> float:
        """What this team's units have learned, as rating precision net of the prior.

        The unit rating's variance is `(S / c)^2 (p^2 + q^2)`; its precision
        less what a pair at the initial deviations would have is what the
        units have earned, and 0 for a team the index never had a play for.
        """
        sides = self._offsets(team)
        scale = self._rating_per_epa**2
        now = 1 / (
            scale
            * (sides.offense.rating_deviation**2 + sides.defense.rating_deviation**2)
        )
        prior = 1 / (
            scale * (self._initial_sd.offense**2 + self._initial_sd.defense**2)
        )
        return max(0.0, now - prior)

    def _blended(self, team: str) -> _Rating:
        """The team's rating for a prediction: record and units, by precision.

        The deviation is the parent's, so the implied-Gaussian prediction's
        spread is the parent's too -- the units sharpen where a team sits,
        not how sure the margin model is of it.
        """
        parent = self.get_rating(team)
        if not self._unit_weight:
            return parent
        earned = self._unit_weight * self.unit_information(team)
        if not earned:
            return parent
        precision = 1 / parent.rating_deviation**2
        rating = (precision * parent.rating + earned * self.unit_rating(team)) / (
            precision + earned
        )
        return _Rating(rating, parent.rating_deviation)

    def matchup_adjustment(self, matchup: Matchup) -> float:
        """The parent's matchup terms, and the talent gap at `talent_edge`.

        In rating points, like the rest of the edge. Nothing when either side
        is unrated -- below FBS, an academy, a season before 2015.
        """
        points = super().matchup_adjustment(matchup)
        if not self._talent_edge:
            return points
        home = self._talent.get(matchup.home, self._season)
        away = self._talent.get(matchup.away, self._season)
        if home is None or away is None:
            return points
        return points + self._talent_edge * (home - away) / self.points_per_rating

    def predict_distribution(self, matchup: Matchup) -> MarginDistribution:
        """The parent's prediction, on blended ratings.

        Written out rather than delegated because the parent reads
        `get_rating` directly; the arithmetic is the parent's, and
        `unit_weight` 0 reproducing it game by game is what guards that.
        """
        home = self._blended(matchup.home)
        away = self._blended(matchup.away)
        edge = self.home_edge(matchup) + self.matchup_adjustment(matchup)
        if not self._implied_prediction:
            return self.gap_distribution(home.rating + edge, away.rating)
        expected = self.points_per_rating * (home.rating + edge - away.rating)
        return NormalMargin(expected, self._margin_sd(home, away), "points")

    def _update_game(self, game: Game) -> Prediction:
        """The parent's step on the margin, then the two contests on EPA.

        The edge is read first, before the parent records the game in the
        rest ledger, and the prediction is this model's, made before either
        step.
        """
        prediction = self.predict_game(game)
        edge = self.home_edge(game) + self.matchup_adjustment(game)
        # The parent's update makes the same prediction again and returns
        # it; the units haven't moved in between, so it is this one.
        super()._update_game(game)
        self._update_sides(game, edge)
        # Only after a game with plays: who started is read off the plays, so
        # an opener ESPN has none for -- a lower-division opponent, most often
        # -- hasn't said who the starter is, and the week-one starter on file
        # is whoever started the next game. Shifting after the opener would
        # hand that game's prediction the name of its own starter.
        if self._game_epa.get(game.game_id) is not None:
            for team in (game.home, game.away):
                if team in self._qb_pending:
                    self._qb_pending.discard(team)
                    self._shift(team, self._quarterback_points(team))
        return prediction

    def _quarterback_points(self, team: str) -> float:
        """What this season's quarterback change is worth to `team`, in points."""
        fact = self._offseason.get(team, self._season)
        if fact is None or not fact.new_quarterback:
            return 0.0
        return self._new_qb_shift + self._qb_quality_shift * fact.quality_change / 0.1

    def _shift(self, team: str, points: float) -> None:
        """Move `team`'s rating and its offense by `points`, now and at preseason.

        Both, because the smoother re-walks the season from the preseason
        ratings: a shift that only touched the current ones would be undone
        by the next pass.
        """
        if not points:
            return
        self._shift_rating(team, points)
        offense_by = points / self._points_per_epa
        for sides in (self._sides, self._sides_preseason):
            if team in sides:
                pair = sides[team]
                sides[team] = pair._replace(
                    offense=pair.offense._replace(
                        rating=pair.offense.rating + offense_by
                    )
                )

    def _widen_for_coach(self, team: str) -> None:
        """A new head coach: the record's deviation and each side's, widened by their own.

        Now and at preseason, for the reason `_shift` moves both. The
        record stays under its `initial_rd`, which never binds in practice;
        a side goes past its initial deviation if the change takes it there,
        since that cap is a fitted clock (the NFL's defense sits on it all
        season) rather than a bound on what a new staff can do. The weekly
        widening never narrows it back, so it lasts until the side's games
        shrink it.
        """
        if self._coach_change_rd:
            for ratings in (self._ratings, self._preseason):
                if team in ratings:
                    current = ratings[team]
                    ratings[team] = current._replace(
                        rating_deviation=min(
                            self._initial_rd,
                            math.hypot(current.rating_deviation, self._coach_change_rd),
                        )
                    )
        offense, defense = self._coach_change_sd
        if not offense and not defense:
            return
        for sides in (self._sides, self._sides_preseason):
            if team in sides:
                pair = sides[team]
                sides[team] = _Sides(
                    pair.offense._replace(
                        rating_deviation=math.hypot(pair.offense.rating_deviation, offense)
                    ),
                    pair.defense._replace(
                        rating_deviation=math.hypot(pair.defense.rating_deviation, defense)
                    ),
                )

    def _shift_rating(self, team: str, points: float) -> None:
        """Move `team`'s rating by `points`, now and at preseason, and not its units."""
        by = points / self.points_per_rating
        for ratings in (self._ratings, self._preseason):
            if team in ratings:
                current = ratings[team]
                ratings[team] = current._replace(rating=current.rating + by)

    def _update_sides(self, game: Game, edge: float) -> None:
        """Both contests, from the units as they stood before the game.

        The two contests touch disjoint units -- home offense and away
        defense, away offense and home defense -- so stepping them one after
        the other is stepping them together.
        """
        epa = self._game_epa.get(game.game_id)
        if epa is None or not epa.home_plays or not epa.away_plays:
            return
        home_reading, away_reading = self._reading(epa, self._success_weight)
        # Read before this game joins it: a contest is measured against the
        # league as it stood, not partly against itself.
        center = self.epa_center
        if self._defense_success_weight == self._success_weight:
            home_against = away_against = None
            self._center_sum += home_reading + away_reading
        else:
            home_against, away_against = self._reading(epa, self._defense_success_weight)
            self._center_sum += (home_reading + away_reading + home_against + away_against) / 2
        self._center_count += 2
        per_play = self.points_per_rating / (2 * self._points_per_epa)
        gap = edge + self._anchor_epa_scale * (self.anchor(game.home) - self.anchor(game.away))
        for offense, defense, reading, against, snaps, sign in (
            (game.home, game.away, home_reading, home_against, epa.home_plays, 1.0),
            (game.away, game.home, away_reading, away_against, epa.away_plays, -1.0),
        ):
            contest = _Contest(
                offense=offense,
                defense=defense,
                epa=reading,
                noise=self._play_sd**2 / snaps,
                base=center + sign * gap * per_play,
                defense_epa=against,
            )
            self._contests_this_week.append(contest)
            attack, guard = self._offsets(offense), self._offsets(defense)
            new_offense, new_defense = _contest_step(
                attack.offense, guard.defense, contest
            )
            self._sides[offense] = attack._replace(offense=new_offense)
            self._sides[defense] = self._offsets(defense)._replace(defense=new_defense)
        if self._success_rates.enabled:
            self._success_rates.add_game(
                game,
                epa,
                gap * per_play / SUCCESS_SCALE[self._league][UNWEIGHTED][0],
                (self.anchor(game.home), self.anchor(game.away)),
            )

    def _reading(self, epa: GameEpa, weight: float) -> tuple[float, float]:
        """What each offense read in this game, home then away, on EPA's scale.

        EPA per play at `weight` 0, success rate on EPA's scale at 1, and
        the blend between: `success_weight` for what the offenses step on,
        `defense_success_weight` for the defenses. Over EPA's snap count either way, so a
        game's noise is priced as it always was. A game with no success
        rate -- an index swept before it carried one, or a weighted one
        whose every snap came with the game decided -- reads its EPA alone.
        """
        if not weight:
            return epa.home, epa.away
        if self._success_reading == WEIGHTED:
            home, away = epa.home_success_weighted, epa.away_success_weighted
        else:
            home, away = epa.home_success, epa.away_success
        if home is None or away is None:
            return epa.home, epa.away
        scale, level = SUCCESS_SCALE[self._league][self._success_reading]
        return (
            (1 - weight) * epa.home + weight * (scale * home + level),
            (1 - weight) * epa.away + weight * (scale * away + level),
        )

    def pass_week(self) -> None:
        """The units' week, then the parent's, whose smoother re-walks both."""
        self._contest_weeks.append(self._contests_this_week)
        self._contests_this_week = []
        self._sides = self._aged_sides(self._sides, self._weekly_sd_increase)
        if self._success_rates.enabled:
            self._success_rates.pass_week()
        super().pass_week()

    def _aged_sides(
        self, sides: Mapping[str, _Sides], increase: "_Pair"
    ) -> dict[str, _Sides]:
        """Each side's deviation widened by its own increase, capped at its own initial.

        A side whose increase is 0 is left as it is rather than rebuilt: the
        defense's weekly increase is 0 by default, and this runs for every
        rated team every week of every smoothing pass.
        """
        offense, defense = increase
        return {
            team: _Sides(
                _widened(pair.offense, offense, self._initial_sd.offense)
                if offense
                else pair.offense,
                _widened(pair.defense, defense, self._initial_sd.defense)
                if defense
                else pair.defense,
            )
            for team, pair in sides.items()
        }

    def _smooth(self) -> None:
        """The parent's sweep, then the same sweep over the units.

        Each unit re-walked from preseason through its contests, the unit
        across from it held at the mean the previous pass ended with and its
        replayed deviation -- `GlickoPredictor._smooth`'s Gauss-Seidel,
        one level down. Only the means come out of it.

        Only the teams with a contest this season are replayed. Every other
        team's mean is its preseason one either way, and carrying the whole
        league's history through every week of every pass was most of what
        the smoother cost.
        """
        super()._smooth()
        played = {
            team
            for week in self._contest_weeks
            for contest in week
            for team in (contest.offense, contest.defense)
        }
        if not played:
            return
        settled = self._sides
        unseen = _Sides(
            _Rating(0.0, self._initial_sd.offense),
            _Rating(0.0, self._initial_sd.defense),
        )
        replay = {team: self._sides_preseason.get(team, unseen) for team in played}

        for week in self._contest_weeks:
            for contest in week:
                attack, guard = replay[contest.offense], replay[contest.defense]
                held_defense = _Rating(
                    settled[contest.defense].defense.rating,
                    guard.defense.rating_deviation,
                )
                held_offense = _Rating(
                    settled[contest.offense].offense.rating,
                    attack.offense.rating_deviation,
                )
                new_offense, _ = _contest_step(attack.offense, held_defense, contest)
                _, new_defense = _contest_step(held_offense, guard.defense, contest)
                replay[contest.offense] = attack._replace(offense=new_offense)
                replay[contest.defense] = replay[contest.defense]._replace(
                    defense=new_defense
                )
            replay = self._aged_sides(replay, self._weekly_sd_increase)

        smoothed = dict(self._sides)
        for team in played:
            pair = self._sides[team]
            smoothed[team] = _Sides(
                pair.offense._replace(rating=replay[team].offense.rating),
                pair.defense._replace(rating=replay[team].defense.rating),
            )
        self._sides = smoothed

    def _roll_over(self) -> None:
        """The parent's offseason, and each side's own.

        An offset regresses toward 0 -- its anchor -- by its side's
        regression, then its deviation grows by its side's increase. The
        center closes the season: its mean becomes next season's prior.
        """
        super()._roll_over()
        keep = _Pair(
            1 - self._unit_regression.offense, 1 - self._unit_regression.defense
        )
        self._sides = self._aged_sides(
            {
                team: _Sides(
                    pair.offense._replace(rating=keep.offense * pair.offense.rating),
                    pair.defense._replace(rating=keep.defense * pair.defense.rating),
                )
                for team, pair in self._sides.items()
            },
            self._season_sd_increase,
        )
        if self._center_count:
            self._center_previous = self._center_sum / self._center_count
        self._center_sum, self._center_count = 0.0, 0
        if self._success_rates.enabled:
            self._success_rates.roll_over()
        self._sides_preseason = dict(self._sides)
        self._contest_weeks = []
        self._contests_this_week = []
        self._qb_pending = set()
        for team, fact in self._offseason.seasons(self._season):
            if fact.coach_departure is not None:
                self._widen_for_coach(team)
            if fact.coach_departure == LEFT_FOR_JOB:
                self._shift(team, self._coach_left_shift)
            if fact.new_quarterback and (self._new_qb_shift or self._qb_quality_shift):
                self._qb_pending.add(team)
        if self._season is None:
            return
        if self._talent_shift:
            for team, level in self._talent.season(self._season):
                change = level - self._talent.previous(team, self._season)
                self._shift(team, self._talent_shift * change)
        # The level every summer, unlike talent: returning production says
        # how this team differs from last season's, which the games have
        # since priced, not a standing level that would pile up. The rating
        # only -- see "What the roster says" for why not the units.
        for weight, index in (
            (self._offense_returning_shift, self._offense_returning),
            (self._defense_returning_shift, self._defense_returning),
        ):
            if weight:
                for team, level in index.season(self._season):
                    self._shift_rating(team, weight * level)

    def state_dict(self) -> dict[str, Any]:
        """The parent's state, the unit knobs, the center and the sides.

        The EPA index stays out, as it does for every play-by-play model.
        """
        return {
            **super().state_dict(),
            "unit_weight": self._unit_weight,
            "points_per_epa": self._points_per_epa,
            "play_sd": self._play_sd,
            "offense_initial_sd": self._initial_sd.offense,
            "defense_initial_sd": self._initial_sd.defense,
            "offense_weekly_sd_increase": self._weekly_sd_increase.offense,
            "defense_weekly_sd_increase": self._weekly_sd_increase.defense,
            "offense_season_sd_increase": self._season_sd_increase.offense,
            "defense_season_sd_increase": self._season_sd_increase.defense,
            "offense_season_regression": self._unit_regression.offense,
            "defense_season_regression": self._unit_regression.defense,
            "coach_left_shift": self._coach_left_shift,
            "new_qb_shift": self._new_qb_shift,
            "qb_quality_shift": self._qb_quality_shift,
            "talent_shift": self._talent_shift,
            "talent_edge": self._talent_edge,
            "offense_returning_shift": self._offense_returning_shift,
            "defense_returning_shift": self._defense_returning_shift,
            "success_weight": self._success_weight,
            "defense_success_weight": self._defense_success_weight,
            "success_reading": self._success_reading,
            "anchor_epa_scale": self._anchor_epa_scale,
            "coach_change_rd": self._coach_change_rd,
            "coach_change_offense_sd": self._coach_change_sd.offense,
            "coach_change_defense_sd": self._coach_change_sd.defense,
            "epa_center_state": [
                self._center_previous,
                self._center_sum,
                self._center_count,
            ],
            "success_rates_state": self._success_rates.state(),
            "sides": {
                team: [
                    pair.offense.rating,
                    pair.offense.rating_deviation,
                    pair.defense.rating,
                    pair.defense.rating_deviation,
                ]
                for team, pair in self._sides.items()
            },
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        params = dict(data)
        params["sides"] = {
            team: _Sides(_Rating(s[0], s[1]), _Rating(s[2], s[3]))
            for team, s in params.pop("sides", {}).items()
        }
        return super().from_state_dict(params)

    @property
    def ratings(self) -> dict[str, Rating]:
        """The parent's ratings, each carrying its two sides where there are any.

        On the team scale, `anchor + 2 (S / c) offset`, so a team's two
        sides average to its unit rating and a 1650 offense reads like a
        1650 team. Teams the index never had a play for carry none.

        Every team carries its `blended` rating, the one
        `predict_distribution` reads -- the parent itself for a team with no
        sides -- and, where its success rates have been read, those; see
        "What a release carries".
        """
        per_rating = (
            self._anchor_epa_scale
            * self.points_per_rating
            / (2 * self._points_per_epa)
            / SUCCESS_SCALE[self._league][UNWEIGHTED][0]
            if self._success_rates.enabled
            else 0.0
        )
        return {
            team: rating._replace(
                units=self._absolute_units(team) if team in self._sides else None,
                blended=self._blended(team).rating,
                success=self._success_rates.rates(team, self.anchor, per_rating)
                if self._success_rates.enabled
                else None,
            )
            for team, rating in super().ratings.items()
        }

    def _absolute_units(self, team: str) -> Units:
        anchor = self.anchor(team)
        scale = 2 * self._rating_per_epa
        pair = self._sides[team]
        return Units(
            Unit(
                anchor + scale * pair.offense.rating,
                scale * pair.offense.rating_deviation,
            ),
            Unit(
                anchor + scale * pair.defense.rating,
                scale * pair.defense.rating_deviation,
            ),
        )

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        """The parent's rebuild, then the sides put back as offsets from this model's anchors."""
        predictor = super().from_ratings(league, ratings, **params)
        scale = 2 * predictor._rating_per_epa
        for team, rating in ratings.items():
            if rating.units is None:
                continue
            anchor = predictor.anchor(team)
            offense, defense = rating.units
            predictor._sides[team] = _Sides(
                _Rating((offense.rating - anchor) / scale, offense.rd / scale),
                _Rating((defense.rating - anchor) / scale, defense.rd / scale),
            )
        return predictor


class _Pair(NamedTuple):
    """One clock setting, per side."""

    offense: float
    defense: float


def _contest_step(
    offense: _Rating, defense: _Rating, contest: _Contest
) -> tuple[_Rating, _Rating]:
    """One contest's Kalman step, both units from the same innovation.

    Each unit's gain is its own variance over the innovation's, which is
    both variances and the reading's noise: the joint step's marginals, the
    cross-covariance it would build dropped as Glicko drops it. A contest
    with a reading of its own for the defense steps each unit on its own
    innovation, against the same expectation.
    """
    p2 = offense.rating_deviation**2
    q2 = defense.rating_deviation**2
    total = p2 + q2 + contest.noise
    expected = contest.base + offense.rating - defense.rating
    innovation = contest.epa - expected
    against = innovation if contest.defense_epa is None else contest.defense_epa - expected
    return (
        _Rating(
            offense.rating + p2 / total * innovation,
            math.sqrt(p2 * (1 - p2 / total)),
        ),
        _Rating(
            defense.rating - q2 / total * against,
            math.sqrt(q2 * (1 - q2 / total)),
        ),
    )


def _widened(unit: _Rating, increase: float, cap: float) -> _Rating:
    """Grown by `increase` up to `cap`, and never narrowed: a side a new
    coach put past its cap stays where it is until its games shrink it."""
    return _Rating(
        unit.rating,
        min(
            max(cap, unit.rating_deviation),
            math.sqrt(unit.rating_deviation**2 + increase**2),
        ),
    )


def _carried(unit: _Rating, clock: RateClock) -> _Rating:
    """A displayed side over the summer: its offset kept at `carry`, its
    deviation refilled toward the spread by what was forgotten."""
    return _Rating(
        clock.carry * unit.rating,
        math.sqrt(
            clock.carry**2 * unit.rating_deviation**2
            + (1 - clock.carry**2) * clock.initial_sd**2
        ),
    )


class _SuccessRates:
    """The displayed success rate: a pair of sides per team on the flat rate.

    See "What a release carries". Stepped by `add_game` from the units'
    `_update_sides`, aged by `pass_week`, carried over the summer by
    `roll_over`. The offsets are in rate, from the team's anchor-implied
    level; the center is the league's average rate, a running mean of every
    side read this season with last season's as a prior, as `epa_center` is.

    "An average opponent" is the average side of the league's upper tier:
    the mean anchor, by contests, of the sides at or above the mean anchor of
    every side read, ever. On ncaafb that is the FBS -- the index also reads
    FCS games, and an opponent averaged over both puts every FBS offense
    eight points of success rate above the rate it posts against its own --
    and on a league without tiers it is every side. Anchors don't move, so
    the count of sides read at each is all it takes.

    A gap -- home edge and anchors -- goes into rate through the units'
    `points_per_epa` and `SUCCESS_SCALE`, and then through a slope measured as
    the replay goes: each game's rate difference on its expected one, from
    1 with `GAP_SLOPE_PRIOR` behind it. A success rate moves less across the
    anchors than EPA does -- on ncaafb, at the success model's 87 points per
    EPA, 0.45 of what the conversion says, at every size of gap -- and taken
    at face value the conversion put FBS offenses three points under their
    anchors and FCS ones near 10%.
    """

    def __init__(self, league: str, state: Mapping[str, Any] | None) -> None:
        self._clocks = SUCCESS_RATE_CLOCKS.get(league)
        state = state or {}
        self._sides: dict[str, _Sides] = {
            team: _Sides(_Rating(s[0], s[1]), _Rating(s[2], s[3]))
            for team, s in state.get("sides", {}).items()
        }
        previous, total, count = state.get("center", (None, 0.0, 0))
        self._previous = None if previous is None else float(previous)
        self._sum, self._count = float(total), int(count)
        self._anchors: dict[float, int] = {
            float(anchor): int(count) for anchor, count in state.get("anchors", [])
        }
        xy, xx = state.get("gap", (0.0, 0.0))
        self._gap_xy, self._gap_xx = float(xy), float(xx)

    @property
    def gap_slope(self) -> float:
        """How far a rate moves per unit of the gap the conversion expects."""
        return (self._gap_xy + GAP_SLOPE_PRIOR) / (self._gap_xx + GAP_SLOPE_PRIOR)

    @property
    def enabled(self) -> bool:
        """Whether this league has clocks to keep the rates on."""
        return self._clocks is not None

    @property
    def center(self) -> float | None:
        """The league's average rate this season; None before any side is read."""
        if self._previous is None:
            return self._sum / self._count if self._count else None
        return (self._previous * CENTER_PRIOR_SIDES + self._sum) / (
            CENTER_PRIOR_SIDES + self._count
        )

    def _offsets(self, team: str) -> _Sides:
        assert self._clocks is not None
        offense, defense = self._clocks
        return self._sides.get(
            team,
            _Sides(_Rating(0.0, offense.initial_sd), _Rating(0.0, defense.initial_sd)),
        )

    def add_game(
        self,
        game: Game,
        epa: GameEpa,
        shift: float,
        anchors: tuple[float, float],
    ) -> None:
        """Both contests, against the center before the game joins it.

        `shift` is what the home offense is expected to read above the
        center, in rate -- the home edge and the anchor gap -- and the away
        offense the same below it. A game without both rates, or the downs
        under them, has nothing to say.
        """
        if (
            epa.home_success is None
            or epa.away_success is None
            or not epa.home_success_plays
            or not epa.away_success_plays
        ):
            return
        center = self.center
        expected = self.gap_slope * shift
        if center is not None:
            p = min(max(center, _RATE_FLOOR), 1 - _RATE_FLOOR)
            for offense, defense, rate, downs, sign in (
                (game.home, game.away, epa.home_success, epa.home_success_plays, 1.0),
                (game.away, game.home, epa.away_success, epa.away_success_plays, -1.0),
            ):
                contest = _Contest(
                    offense=offense,
                    defense=defense,
                    epa=rate,
                    noise=p * (1 - p) / downs,
                    base=center + sign * expected,
                )
                attack = self._offsets(offense)
                new_offense, new_defense = _contest_step(
                    attack.offense, self._offsets(defense).defense, contest
                )
                self._sides[offense] = attack._replace(offense=new_offense)
                self._sides[defense] = self._offsets(defense)._replace(
                    defense=new_defense
                )
            self._gap_xy += shift * (epa.home_success - epa.away_success) / 2
            self._gap_xx += shift**2
        self._sum += epa.home_success + epa.away_success
        self._count += 2
        for anchor in anchors:
            self._anchors[anchor] = self._anchors.get(anchor, 0) + 1

    def pass_week(self) -> None:
        assert self._clocks is not None
        offense, defense = self._clocks
        self._sides = {
            team: _Sides(
                _widened(pair.offense, offense.weekly_sd, offense.initial_sd),
                _widened(pair.defense, defense.weekly_sd, defense.initial_sd),
            )
            for team, pair in self._sides.items()
        }

    def roll_over(self) -> None:
        assert self._clocks is not None
        offense, defense = self._clocks
        self._sides = {
            team: _Sides(_carried(pair.offense, offense), _carried(pair.defense, defense))
            for team, pair in self._sides.items()
        }
        if self._count:
            self._previous = self._sum / self._count
        self._sum, self._count = 0.0, 0

    def rates(
        self, team: str, anchor_of: Callable[[str], float], per_rating: float
    ) -> SuccessRates | None:
        """`team`'s offense against an average defense, and its defense
        against an average offense, on a neutral field; None for a team
        never read. `per_rating` is rate per rating point of anchor gap.

        The anchor is asked for only for a team that has been read, as the
        units ask only for teams with sides.
        """
        center = self.center
        if team not in self._sides or center is None:
            return None
        anchor = anchor_of(team)
        level = self.gap_slope * (anchor - self._reference(anchor)) * per_rating
        pair = self._sides[team]
        return SuccessRates(
            Rate(_as_rate(center + level + pair.offense.rating), pair.offense.rating_deviation),
            Rate(_as_rate(center - level - pair.defense.rating), pair.defense.rating_deviation),
        )

    def _reference(self, fallback: float) -> float:
        """The anchor of an average opponent: the upper tier's mean side."""
        if not self._anchors:
            return fallback
        count = sum(self._anchors.values())
        mean = sum(a * n for a, n in self._anchors.items()) / count
        # The tier's floor sits a hair under the mean: with every side at one
        # anchor, as on a league without tiers, the mean can round above it.
        floor = mean - 1e-9 * max(1.0, abs(mean))
        upper = [(a, n) for a, n in self._anchors.items() if a >= floor]
        return sum(a * n for a, n in upper) / sum(n for _, n in upper)

    def state(self) -> dict[str, Any]:
        return {
            "sides": {
                team: [
                    pair.offense.rating,
                    pair.offense.rating_deviation,
                    pair.defense.rating,
                    pair.defense.rating_deviation,
                ]
                for team, pair in self._sides.items()
            },
            "center": [self._previous, self._sum, self._count],
            "anchors": [[anchor, count] for anchor, count in self._anchors.items()],
            "gap": [self._gap_xy, self._gap_xx],
        }


def _as_rate(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _non_negative(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value

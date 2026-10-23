# Success rate by kind of play and by situation

2026-10-23. Does the success rate the units read get better split into
dropbacks and runs, by offense and defense -- or into the obvious situations
(third and long, short yardage, running out the clock) -- or with
interactions between the two sides' extremes? Everything is scored in brier
1e-4 or squared points of margin, fit through 2021 (ncaafb) or 2017 (the
NFL's raw checks) and held out after.

## What a dropback is

`kinds.py`: a pass of any kind, a sack, an interception, and a "Rush" whose
text says the quarterback scrambled; a fumble or safety goes with the kind
its text names. On the NFL 5% of "Rush" plays are scrambles. **The college
feed never says so** (0.00% of ncaafb runs): there, every scramble is a run.

| | dropbacks | dropback success | run success |
|---|---|---|---|
| NFL 2006-2025 | 60% | 48% | 39% |
| ncaafb 2014-2026 | 48% | 44.7% (neutral downs) | 40.7% (neutral downs) |

Situations (`nfl_situations.py`, `ncaafb_cells.py`) read down, distance and
the offense's win probability, never the call:

| situation | NFL share | NFL dropbacks | NFL pass / run SR | ncaafb share | ncaafb dropbacks | ncaafb pass / run SR |
|---|---|---|---|---|---|---|
| long: 3rd/4th and 7+, 2nd and 12+ | 10% | 89% | 38 / 25 | 11% | 75% | 33 / 32 |
| short: 3rd/4th and 2 or less, 2nd and 1 | 5% | 36% | 55 / 52 | 6% | 24% | 51 / 60 |
| clock: 4th quarter, win prob > 0.8 | 8% | 34% | 48 / 35 | 9% | 25% | 44 / 41 |
| hurry: 4th quarter, win prob < 0.2 | 10% | 83% | 48 / 46 | 10% | 64% | 39 / 43 |
| two minutes before the half | 7% | 80% | 48 / 45 | 5% | 62% | 43 / 46 |
| neutral | 60% | 54% | 51 / 38 | 60% | 45% | 45 / 41 |

## Rates as features (no filter)

**NFL** (`nfl_split.py`, `nfl_cells.py`): raw season-to-date rates, shrunk
by 30 downs, for games once both teams have 4; margins 2018-2025 against
the pooled rate:

| reading | change, pts^2 | t |
|---|---|---|
| fixed mix, 0.65 dropbacks | -1.86 | -3.2 |
| free pass/run split | -1.80 | -2.9 |
| neutral downs only | +3.10 | +2.2 |
| split + third-and-long passing + short-yardage running | -0.73 | -0.7 |
| split + each team's own dropback share | -2.60 | -2.1 |

Pass rates are the more reliable (offense 0.82 vs 0.64 within a season,
0.64 vs 0.49 year to year). Interaction products -- pass offense by pass
defense, run by run, a pass-heavy offense against a pass-funnel defense --
add nothing. The pooled rate's flaw is that it credits calling dropbacks,
which trailing teams do. Beside the success model's own win probability,
every raw rate costs (+4 to +18), the situational cells +0.0.

**ncaafb** (`ncaafb_cells.py`): raw rates there would measure the schedule,
so each cell is a ridge refit before every week on the season's earlier
contests -- center, home, offense, defense, each team pulled toward half of
where it ended last season by K downs. FBS v FBS from a side's fifth game,
held out 2022-2026, change against the pooled rate in pts^2:

| reading | K 60 | K 150 | K 400 |
|---|---|---|---|
| fixed mix 0.48 | +0.0 | +0.2 | +2.1 (t 2.4) |
| split pass/run | +1.2 | +1.2 | +2.8 |
| neutral downs only | +12.9 (t 4.7) | +12.8 | +14.8 |
| split + third-and-long passing + short-yardage running | +1.5 | +1.4 | +3.2 |
| split + clock runs + hurry passes | +2.8 | +3.2 | +5.0 |

Nothing, and for a reason: college dropbacks succeed only 4 points more
often than runs on neutral downs (the NFL's gap is 12), so the mix moves a
pooled rate little. Beside the success model: -1.4 to +9.8, no |t| above 1.9.

## In the units (the filter is the opponent adjustment)

`nfl_index.py` and `ncaafb_index.py` rebuild each league's swept index with
the mixed rate in the success slot, at its own EPA-scale constants
(measured as `SUCCESS_SCALE` was; the pooled rate rebuilt from the plays
matches the index on every game and reproduces the published constants).
The replays then run the published success models unchanged -- the units'
contest against the opponent's unit, the anchors, the smoother's passes
(2 NFL, 3 ncaafb), the roster priors -- on the pinned data. `nfl_replay.py`
for the NFL, `ncaafb_run.py` and `../ncaafb/search.py` through `mixed.sh`
for ncaafb; `compare.py` and `where.py` pair them.

**NFL**, all games, against the pooled reading given the same refit
(success weight, `points_per_epa`, `play_sd` through 2021):

| mix | <=2021 | held out | 2014 on | t |
|---|---|---|---|---|
| 0.60 | -0.4 | +1.0 | +0.1 | +0.1 |
| 0.65 | -0.1 | +1.3 | +0.4 | +0.5 |

Nothing: what the raw rates gain, the filter's opponent adjustment,
carryover and margin record already have.

**ncaafb**, FBS v FBS, at the published success model's knobs:

| mix | <=2021 | held out | 2014 on | t | games 1-4 | t | later |
|---|---|---|---|---|---|---|---|
| 0.50 | -0.44 | -0.26 (4/5) | -0.38 (10/13) | -2.5 | -1.08 | -3.2 | -0.06 |
| 0.60 | -0.58 | -0.36 (3/5) | -0.50 (9/13) | -2.2 | -1.58 | -3.1 | -0.01 |
| weighted pooled | -0.27 | -0.49 | -0.35 | -0.7 | -2.00 | -1.7 | +0.40 |
| weighted, mix 0.60 | -0.59 | -0.52 | -0.56 | -1.1 | -2.77 | -2.2 | +0.43 |

Small and steady, and all of it early (`where.py`): -2.1 (t -3.3) in a
side's games 2-4, -2.3 (t -2.6) when either side's last game was against a
lower tier, -1.2 (t -3.0) after a 28-point game, about 0 otherwise. An FBS
team that ran out the clock on an FCS opponent read low on the pooled rate.
The competitiveness weight goes after the same thing more noisily, and the
two don't add.

Refit (`../ncaafb/search.py`: the success weight, `points_per_epa` and
`play_sd` through 2021, from the published knobs) for each reading, against
the pooled reading's own refit -- which stays on the published knobs:

| mix | success weight | <=2021 | held out | 2014 on | t |
|---|---|---|---|---|---|
| pooled | 0.59 | | | | |
| 0.50 | 0.74 | -0.50 | -0.33 (3/5) | -0.43 (8/13) | -1.5 |
| 0.60 | 0.81 | -0.70 | -0.47 (3/5) | -0.61 (8/13) | -1.4 |
| 0.60, defense free | 0.96, defense 1.0 | -1.89 | -0.30 (3/5) | -1.30 | -1.6 |

The refits trust the mixed rate more than the pooled one (0.81 against
0.59), and the gain stays the same size, spread a little later into the
season. Opening the defense does what it did on the pooled rate: wins
through 2021, not held out. Leave it on EPA.

## The split, for a reader

Shipped at zero weight: the sweep counts each side's dropbacks and runs
(`cassandra.epa_build.is_dropback`), and every units model keeps a pair of
sides per team for each kind beside the every-down pair, published as
`success.dropbacks` and `success.runs`. Predictions never read them.

Clocks (`kind_clocks.py`): each kind's opponent-adjusted true spread and
carry, measured as `combine/persistence.py` measured the every-down rate --
whose rows this reproduces (NFL 0.027 / 0.58 offense, 0.019 / 0.60 defense;
ncaafb 0.048 / 0.85, 0.043 / 0.85). The carry is the season-to-season
correlation over the season's reliability.

| | dropbacks, offense | dropbacks, defense | runs, offense | runs, defense |
|---|---|---|---|---|
| NFL | 0.029 / 0.67 | 0.020 / 0.62 | 0.027 / 0.65 | 0.022 / 0.75 |
| ncaafb | 0.047 / 0.80 | 0.037 / 0.85 | 0.043 / 0.85 | 0.041 / 0.79 |

Checked end to end (`sweep_split.py`, `split_replay.py`): the production
sweep over both leagues into a private index, every earlier reading as the
harness's index had it (the weighted success rate to 1e-9), every game's
split adding back to its flat rate; the published success models replayed
on it predict every game to the bit as without it, and show every team the
same every-down rates. What they show:

| | dropback offense | run offense | spread across teams | dropback vs run, r |
|---|---|---|---|---|
| NFL (all seasons' end) | 48.7% | 40.8% | 1.5 / 1.5 | 0.53 offense, 0.56 defense |
| ncaafb, FBS 2026 | 42.7% | 42.2% | 4.1 / 3.4 | 0.76 offense, 0.85 defense |

Rated apart, each kind is pulled harder toward its anchor's level than the
every-down rate is, having half the downs: on ncaafb a top-fifth offense's
every-down rate is 0.9 points above the league mix of its kinds (Georgia:
51.1% every down, 50.8% dropbacks, 49.0% runs), a bottom-fifth one 0.1
under. On the NFL the gap is a flat 0.2-0.4, the mix's weights.

## Verdict

- Situational cells, neutral-only rates and interactions: nothing, either
  league, raw or opponent-adjusted. Don't build.
- A fixed pass/run mix: nothing in the NFL's units. On ncaafb about -0.5
  on FBS v FBS (t -1.4 to -2.5 by setting, held out -0.3 to -0.5), from
  the early-season games after a mismatch. Small; worth building only as
  a cheap reading (`success_reading` "mixed", ncaafb, the sweep counting
  each side's dropbacks and runs), not as new state.

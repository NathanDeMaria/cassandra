# NFL success rate: team betas, and a reading for the units

2026-10-12 to 2026-10-13. A play is a success when its raw EPA is positive.
Over an offense's dropbacks and runs -- no punts, kicks, penalties-only
snaps, kneels or spikes -- that is a rate with a binomial count behind it,
so a team's success rate is a beta and a game is a beta-binomial draw. This
is what that model found, and how the same number went into
`UnitMarginGlickoPredictor` as `success_weight`.

## Team betas, and what happens when they meet

`success.py`, `examples.py`: each offense and defense as a beta posterior
(league rate as the prior mean, κ plays of prior weight).
`collide.py` (tests in `collide_test.py`): when offense meets defense, a
snap succeeds with

    logit p = logit o + logit d - λ ± home

-- the odds-ratio rule, so an average pair succeeds at the league's rate.
The posterior over both teams after a game is computed on a grid and
projected back onto two betas by their moments. Teams are stored as logit
deviations recentred weekly, so λ absorbs the league's drift (40.5% in
2006, 49.5% in 2021, 46.4% in 2025). The offseason is AR(1): each side
keeps `carry` of its deviation and refills its variance toward the talent
spread.

`fit.py` fits the six knobs by walk-forward predictive likelihood on
2007-2025 (2006 is burn-in), every game scored before it updates anything:

| knob | offense | defense |
|---|---|---|
| κ (plays of prior weight) | 246 (sd 3.2 points of success rate) | 512 (sd 2.2) |
| carryover | 0.65 | 0.61 |

and κ_game 1030 (a 63-snap game behaves like 59 independent snaps), home
0.027 logit. Against the league rate alone: +0.094 log-likelihood per
meeting, RMSE 6.88 vs 7.40 points of success rate; on 2026's first five
weeks, +0.106 (+0.045 without the carryover). The offense carries most of
it (+0.067 alone, the defense +0.021). Calibrated within 0.4 points by
decile.

`drift_eval.py`: a weekly variance increase, for injuries, fits 0.0037 a
week on both sides. Worth +0.0009 per meeting overall, +0.0042 on offenses
whose quarterback the qb_out index flags out, +0.0062 on the second game
running he's out -- and still leaves a 1.5-point bias on backup-quarterback
offenses, which a flag would fix and a drift can't.

## As the units' reading

`units/`: the published NFL `glicko_margin_units_offseason` replayed
walk-forward with its units rated on success rate instead of EPA per play,
or a blend. Success rate goes onto EPA's scale first by the ratio of the
two same-game margin slopes (`readings.py`, `SUCCESS_SCALE` in the
predictor), so every knob keeps its units. `search.py` refits the eight
knobs in the reading's units by Nelder-Mead on 1980-2021 for each blend
weight, and gives EPA the same refit. 2022-2026 is held out.

Results are in `cassandra/predictor/unit_glicko.py`'s "Success rate" and
"A new head coach" sections. In short, in brier 1e-4:

- 75% success rate with the offense forgetting 0.02 a week: -8.5 against
  the published model on 2014-2026 (t -2.1, better in 11 seasons of 13
  and all 5 held out), -5.3 against EPA given the same forgetting.
- Success rate and the forgetting each carry about half; neither alone
  clears t 1.6. The forgetting pays most on games the qb_out index flags.
- Adding success rate from 2006-2013 wins in-sample and loses held out.
- The coach-change deviations, refit jointly with everything else
  (`prod.py full`): the record's at 62 rating points, the sides' at 0,
  -1.0 on 2006-2026 against the same fit without them (t -0.9). Fit
  alone, the offense's deviation goes to where a new coach's first game
  rewrites it, and games 1-4 pay +6.3.
- The units do worse than no units on games a quarterback is flagged out,
  which is why the config searches `qb_out_penalty` too.

The joint refit and the 0.75 refit are the first two seeds in
`models/nfl/glicko_margin_units_success.json`; the third is the published
model at weight 0. `units/seedcheck.py` replays them through the
production code: train 0.219292 / 0.219355 / 0.219636, held out
0.217756 / 0.217874 / 0.218827.

## ncaafb

`ncaafb/`: the published `glicko_margin_units_roster` replayed the same
way, scored as its search is, on FBS v FBS brier. `replay.py` resumes every
probe from a snapshot at the end of 2013 -- the units have no reading
before 2014, so nothing earlier depends on a reading knob -- and stops at
2021 for a search, about a third of the games: 13s a probe against 40s,
and identical to the full replay to 1e-15. The snapshot needs the season
clock and the pending quarterback shifts copied over by hand, which a
predictor's `state_dict` doesn't carry.

Every replay reads a pinned copy of `~/.cassandra/predictor/data`
(`pin.py`). Another session refreshed the shared directory from s3 between
two refits started 14 minutes apart, which moved 2014-on win probabilities
by up to 0.027; the numbers here are all re-scored on the copy
(`rescore.py`), which moved them by 0.1-0.3e-4.

**Does success rate break in mismatches?** (`mismatch.py`) No. Per side it
runs 26-57% (5th-95th), far from either bound. Fit offense + defense per
season leaving each week out, it is as additive as EPA (observed on
predicted slope 0.958 against EPA's 0.960) and fits worse on logits.

**Garbage time** (`sweep_weighted.py`): each snap weighted by the
competitiveness weight behind the weighted EPA. Its own EPA scale is 1.494
(the unweighted rate's 1.798), and at that scale it is even with the
unweighted rate on ncaafb (-0.2e-4, t -0.3) and worse on the NFL (held out
-5.6 against the published model where the unweighted gains -9.5).

**Refits** (`search.py`, the eight reading knobs and `unit_weight` by
Nelder-Mead on FBS brier through 2021; `compare.py` for the slices), in
1e-4 against the EPA refit:

| reading | 2014-2026 | t | held out 2022-2026 | t |
|---|---|---|---|---|
| unweighted success rate | -1.6 | -0.6 | +2.9 | +0.7 |
| weighted success rate | -1.8 | -0.7 | +3.1 | +0.7 |
| weighted success rate + weighted EPA | -2.6 | -0.9 | +2.5 | +0.6 |
| weighted EPA, no success rate | 0.0 | 0.0 | -4.3 | -2.0 |

Every success-rate fit goes to weight 1. One season alone moves by 20
(2023: +13 to +19), so none of this is distinguishable from nothing. The
held-out loss is the refit's clocks more than its reading: at the
unweighted fit's knobs, EPA itself is 5.6 worse held out.

**Offense and defense apart** (`defense_success_weight`). At the EPA
refit's knobs, success rate on the offense alone (0.5) gains 1.4 on
2014-2026 (t -1.5, 9 seasons of 13) and 1.4 held out, 2.5 after a team's
fourth game (t -2.7) and 3.4 in Group of Five games (t -2.5), and costs
1.0 in games 1-4; on the defense alone it gains 0.1 and loses 1.7 held
out. On the NFL, at the joint refit's knobs, the offense's weight from 0
to 1 is worth -12.5 on 2014-2026 and the defense's 0.3 at most, though the
NFL fits hold their defenses nearly still.

Refit with only the weights, `points_per_epa` and `play_sd` free, in 1e-4
against the published roster model, which saw 2022-2026 in its own search:

| run | weights (offense, defense) | 2014-2026 | t | held out | t |
|---|---|---|---|---|---|
| `nc_c_off`, EPA refit's clocks | 0.70, 0 | -3.2 | -1.3 | -0.3 | -0.1 |
| `nc_c_off_w`, the weighted rate | 0.61, 0 | -3.3 | -1.4 | -0.7 | -0.2 |
| `nc_c_split` | 0.88, 1.0 | -4.6 | -1.8 | +0.4 | +0.1 |
| `nc_p_epa`, published clocks, `unit_weight` free | EPA | -1.2 | -2.3 | -1.4 | -1.7 |
| `nc_p_off`, the same | 0.59, 0 | -2.0 | -2.2 | -2.4 | -1.6 |
| `nc_p_epa_uw`, `unit_weight` uncapped (5.0) | EPA | -1.7 | -0.8 | +0.7 | +0.2 |
| `nc_p_off_uw`, the same (5.95) | 0.78, 0 | -3.7 | -1.4 | -1.5 | -0.4 |
| run 20261014-010003, every season in-sample | 0.54, 0 | -3.3 | -2.2 | -4.2 | -1.7 |

`nc_p_off` against `nc_p_epa` is what success rate adds on the published
clocks: -0.9 (t -1.0), -1.0 held out, -1.5 after week 4 (t -1.9), -3.4 in
Group of Five games (t -2.5). The NFL's split refits (`units/prod.py
split`) send the defense's weight to 0 and gain at most 1.5 held out
(t -0.9) against the joint refit.

`unit_weight` above 1 is where the fits overfit: the published clocks' refits
stop at the replay's old cap of 1, and uncapped (`search.py` now searches it
as a positive knob) they run to 5-6, win through 2021 and lose held out.
The Batch search over every season, boxed at 2, stopped at 1.92 (`rescore.py
batch_nc_success` replays it, 0.174148 against the run's 0.174144). Its NFL
twin is the joint refit itself -- 600 probes never beat that seed -- so its
-10.7 on 2022-2026 against the published model (t -1.5, 4 seasons of 5) is
held out.

**Strength of schedule** (`anchors.py`, `bias.py`). ncaafb's anchors are one
rating per conference, so the conference's level is the anchor and the
games across conferences. A reading that moves less across an anchor gap
than `points_per_epa` converts it to would pull the stronger conferences'
sides down in every such game, and conference play would never put them
back. Measured as the slope of an offense's surprise on the anchor gap, in
FBS games across conferences: -0.25 under the published model (t -15;
power sides 8-12 points under the Group of Five's at season end), +0.04
under the EPA refit, -0.06 and -0.09 under the success-rate refits. But
the success-rate refits overrate Group of Five teams no more than EPA's
does (power side's outcome less prediction -0.6 to -1.1 points of win
probability, se 0.9, EPA's -1.1), the held-out loss is not on teams that
changed conference, and `anchor_epa_scale` at 0.75 to 1.05 moves brier
by under 1e-4 and the slope away from 0. The slope is not only the
conversion.

## Running it

Data lives under `~/.cassandra/success_rate/`, never beside the scripts.

    python extract_plays.py            # every snap with raw EPA, to nfl_plays.parquet
    python fit.py fit                  # the collision's knobs, to collide_params.json
    python examples.py                 # figures, to fig/
    python collide_examples.py

    cd units
    python prep.py                     # NFL seasons, pickled once
    python sweep_index.py nfl          # the EPA index with success rates, in memory
    python search.py fit_b75 0.75 2014 # one blend's refit
    python summarize.py; python ablate.py
    python prod.py check               # production code reproduces the harness
    python prod.py full full_b75 fit_b75
    python coach_eval.py full_b75:fit_b75

    cd ../ncaafb                       # after units/sweep_index.py ncaafb
    python replay.py prep              # seasons from s3, priors from the replay cache
    python replay.py run published     # one replay; knob=value to move knobs
    python mismatch.py
    python sweep_weighted.py ncaafb    # the weighted rate, into its own pickle
    python pin.py                      # (from ..) snapshot the shared predictor data, once
    python search.py nc_fit_epa --set success_weight=0
    python search.py nc_c_off --from nc_fit_epa --knobs success_weight,points_per_epa,play_sd \
        --set success_weight=0.5 defense_success_weight=0
    python rescore.py nc_fit_epa       # a run's knobs again on the pinned data, as NAME_p
    python compare.py nc_fit_free:nc_fit_epa
    python anchors.py record nc_fit_epa; python anchors.py report nc_fit_epa
    python bias.py

The refits of the weighted rate read `sweep_weighted.py`'s pickle with
`NCAAFB_INDEX=ncaafb_index_weighted.pkl NCAAFB_SUCCESS_SCALE=1.494,-0.624`,
which puts it where `success_reading="weighted"` does (checked to 1e-15);
`NCAAFB_EPA=weighted` puts the units on the weighted EPA.

Run with `PYTHONPATH` at the checkout whose `cassandra` is meant.
`sweep_index.py` sweeps into a pickle and never writes the shared
`{league}_epa.json`: other sessions read it, and a checkout from before
`EpaFit.success` can't parse the new header.

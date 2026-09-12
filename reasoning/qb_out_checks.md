# The qb_out flag, checked by hand and by residual

2026-09-12. The `{league}_qb_out.json` index says which teams played a game
without the quarterback who had been starting. Nathan doubted it; this is
what a spot check found, what was changed, and how the change was scored.

## What was wrong

Rebuilt the 2025 ncaafb index locally and dumped, for each of the 265
flagged team-games, who was expected, who actually threw, and how many
attempts. Two problems.

**Names with a suffix or a two-word surname didn't parse in the `#N F.Last`
format.** `D.Williams Jr.`, `M.Van Buren Jr.`, `C.Del Rio-Wilson`, `K.Ah
Yat`, `J.French IV`, `A.Barnett III` matched nothing; Washington vs Illinois
had 34 of 35 pass plays unparsed and its "starter" was a receiver who threw
one trick play. 15 teams, 34 team-games, 26 of the 265 flags, all from the
week that format took over (week 10 of 2025; all of 2026).

**"Expected starter = last game's busiest passer" fired on returns.** A
starter hurt in the first quarter throws six passes, his backup thirty; the
backup becomes expected, the real absence next week is *not* flagged, and
the starter's return *is*. 103 of the 265 flags were on a game the team's
season-long starter played.

## Games checked against the record

| game | flag said | what happened | old rule | new rule |
|---|---|---|---|---|
| Colorado–Houston 9/12/25 | Salter out | Salter benched, Staub started | right | right |
| Kentucky–E. Michigan 9/13/25 | Calzada out | Boley took the job | right | right |
| Syracuse–Duke 9/27/25 | Angeli out | Achilles vs Clemson | right | right |
| Oklahoma–Kent St 10/4/25 | Mateer out | hand surgery, Hawkins started | right | right |
| Wisconsin–Michigan 10/4/25 | O'Neil out | hurt; Simmons started | right | right |
| Clemson–SMU 10/18/25 | Klubnik out | ankle; Vizzina started | right | right |
| ASU–Utah 10/11/25 | Leavitt out | out; Jeff Sims started | right | right |
| UCLA–Ohio St 11/15/25 | Iamaleava out | hurt; Duncan started | right | right |
| Hawai'i–Sam Houston 9/7/25 | Alejado out | intended starter, injured (Nathan) | right | right |
| Tulsa–Navy 9/14/25 | Francis out | injured, Hayes took over (Nathan) | right | right |
| NC Central–MVSU 9/23/23 | Richard out | ankle; Harris started ([hbcusports](https://hbcusports.com/2023/09/23/nc-central-overwhelms-mississippi-valley-state-in-circle-city-classic-without-davius-richard/)) | right | right |
| Florida–Samford 9/7/24 | Mertz out | concussion; Lagway started | right | right |
| Stetson–Furman 9/14/24 | Meitz out | O'Connor started; Meitz a 9-game starter ([gohatters](https://gohatters.com/news/2024/9/14/football-hatters-come-up-short-in-a-48-7-loss-to-furman.aspx)) | right | right |
| South Carolina–Missouri 9/20/25 | Doty out | Sellers hurt in Q1 vs Vandy, started vs Mizzou | **wrong** | not flagged |
| Florida–USF 9/6/25 | T. Jones out | Jones was mop-up in a 55-0; Lagway started both | **wrong** | not flagged |
| Colorado–Wyoming 9/21/25 | Staub out | Salter back | **wrong** | not flagged |
| Oklahoma–Texas 10/11/25 | Hawkins out | Mateer back | **wrong** | not flagged |
| Clemson–Duke 11/1/25 | Vizzina out | Klubnik back | **wrong** | not flagged |
| Texas Tech–K-State 11/1/25 | Griffis out | Morton back | **wrong** | not flagged |
| Texas Tech–Arizona St 10/18/25 | *(nothing)* | Morton hurt Q1 vs Kansas, out vs ASU | **missed** | flagged |
| Memphis–USF 10/25/25 | Hill out | Lewis the starter, Hill relief (Nathan) | **wrong** | not flagged |
| Miami (OH)–NIU 10/4/25 | Hesson out | Finn the starter, back after two Hesson starts (Nathan) | **wrong** | not flagged |
| Illinois State–E. Illinois 9/13/25 | Pellant out | Rittenhouse hurt at OU, back week 3 ([prairiestatepigskin](https://prairiestatepigskin.com/2025/08/30/redbird-rewind-rittenhouse-injured-as-fbs-no-18-oklahoma-defeats-illinois-state/)) | **wrong** | not flagged |
| Washington–Illinois 10/25/25 | Williams Jr. out | threw 34 passes, unparsed | **wrong** | not flagged |
| Marshall–JMU 11/8/25 | Del Rio-Wilson out | threw 37, unparsed | **wrong** | not flagged |
| Georgia Southern 11/7/25, JMU 10/29/25, Montana 10/25/25 | starter out | unparsed suffix names | **wrong** | not flagged |
| Marshall–Missouri St 9/6/25 | Long out | three QBs played the opener; DRW the eventual starter (Nathan) | right, uninformative | flagged |
| Arkansas St–Southern Miss 9/23/23 | Dailey out | three-way battle, Raynor won it ([kait8](https://www.kait8.com/2023/09/24/raynors-first-start-arkansas-state-opens-sun-belt-play-with-win-over-southern-miss/)) | competition | flagged |
| App State, 2025 | Swann / Kohl alternating | tight QB battle (Nathan) | competition | flagged each switch |

Competition cases are accepted false positives: the plays cannot tell a
battle from an injury, and Nathan is fine with that if the data backs the
rest up.

## How the rule was chosen

Not by eye. Replay `glicko_full` with `qb_out_penalty=0`, take each flagged
team-game's `margin_residual` from the team's side (positive: the team beat
the model), and compare rule variants by mean residual and by the margin
error a constant shift on the flagged games would recover. The parser fixes
were applied to every variant so only the rule differs.

ncaafb 2023–2025 (11,325 games), NFL 2015–2024 (2,560 games):

| rule | ncaafb n | mean | recovers | nfl n | mean | recovers |
|---|---|---|---|---|---|---|
| last game's starter (as shipped, old parser) | 265 (2025 only) | −1.39 | — | | | |
| last game's starter | 994 | −1.36 | 27 | | | |
| 3-game window of starts | 1288 | −1.02 | 5 | | | |
| senior man, whole absence, handover 3 (main's) | 1582 | −0.75 | 5 | 696 | −2.64 | 136 |
| established man, first game only (handover 1) | 822 | −1.93 | **68** | 332 | −3.02 | 79 |
| established man, whole absence, handover 3 | 1394 | −1.11 | 20 | 628 | −2.73 | **130** |

"Recovers" is margin-points summed over the games, so 68 on 11,325 games
is 0.006 MAE and 130 on 2,560 is 0.05 — a quarterback is worth about ten
times as much to an NFL prediction as to a college one, which is also why
the NFL numbers are far more certain.

The split by whose flag it was is what explains the table. Games where the
usual starter was genuinely out run −2.4 to −3.0 in both leagues. Games
flagged when the usual starter was *playing* — his return, or the second
week of a switch to the man who turned out to be the starter — run +0.75
to +2.5. The NFL has few of those; college has hundreds, so flagging a
whole absence pays there and costs here. Hence `HANDOVER` is per league:
NFL 3, ncaafb 1.

Scripts (session scratch, not checked in): `spot_check.py` rebuilt one
season's parsed games into a pickle, `replay.py` produced the residuals,
`compare_rules.py` scored the variants. About four minutes for a league
replay; a season's plays read in under half a minute.

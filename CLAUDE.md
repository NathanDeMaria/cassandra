# cassandra

Team ratings and win probabilities for ncaafb, nfl, mens and womens college
basketball, nhl and wnba, plus the Kalshi picks and betting tools built on them.

## What belongs where

- **cassandra**: predictions and modelling only. Reading collected data and
  fitting from it belongs here; crawling a source does not.
- **EndGame** (`endgame`, `endgame-aws`): game and season records and the jobs
  that fetch them. Keep `Game` and `Season` light.
- **call-it-what-you-want**: who a team is (names, divisions, conferences).
- **say-youll-remember-me**: offseason facts. **ten-minute-version**: basketball
  player ratings. **the-lucky-ones**: EPA and win probability fits.
- **invisible-string**: the site. A release carries every team the model
  rates; what a reader sees is invisible-string's decision, not `publish.py`'s.

## Commands

```
make check   # what CI runs: ruff format --check, ruff check, ty
make lint    # the same, fixing what it can
make test    # pytest
```

Other targets (`report`, `picks`, `bet`, `autobet`, `ledger`, `publish`,
`submit`, ...) are in `project.mk`. `Makefile`, `ci.yml` and the lower half of
`pyproject.toml` are owned by the mirrorball template; change them there.

## Working in this checkout

Several sessions share this machine and this repository at once.

- Work in your own worktree on your own branch, cut from `origin/main`. Read
  and edit files from that worktree, never from `/workspaces/cassandra`: the
  main checkout sits on an old branch and can be weeks behind.
- `main` moves while you work. `git fetch` before saying a branch has unmerged
  work or opening a pull request for it.
- Never use bare `git stash`, and never `pkill -f "<pattern>"`: both reach
  other sessions. Stop a process by a PID you have checked.
- The environment sets `VIRTUAL_ENV=/workspaces/cassandra/.venv`, shared by
  every session. `uv run` and `uv sync` ignore it; `uv pip` does not, so pass
  `--python <repo>/.venv/bin/python` or unset it first.
- That venv's `cassandra.pth` points at the main checkout, so a script outside
  the repo imports that tree whatever the working directory. Choose the tree
  with `PYTHONPATH=<worktree>` and print `cassandra.__file__` to prove it.
- Format and stage only the files you touched (`ruff format <files>`, `git add
  <files>`), never `git add -A`.
- Inspect a dependency with the `inspect-dependency` skill, called by its
  relative path, not with `python3 -c`. Edit files with the edit tools, not a
  heredoc.

## Pull requests

- The pull request is Nathan's review. Never merge one unless he asks for that
  merge; a go-ahead to build is not one.
- A push that touches `.github/workflows/` must go over the ssh remote. The gh
  token has no `workflow` scope.
- The repository is public. No stakes, limits, thresholds or account balances
  in code, tests, docs, commit messages or pull requests. Betting values live
  only in s3 `cassandra/betting/autobet.json`; code holds the structures.

## Secrets

- The Kalshi key is in `/workspaces/cassandra/.kalshi/` and the CFBD key in
  `cfbdkey.txt`. Both are ignored. Never print, copy or commit them.
- An expired AWS login is handled by the `aws-login` skill.

## Measuring a model change

- Replays read `~/.cassandra/predictor/data/{league}_{Class}_priors.json`,
  which other sessions rewrite. Pass a private `priors_path`, or keep every
  variant you compare in one batch.
- Pickle the seasons once and replay against an empty `OddsDatabase({})` with
  `post_callbacks=False`; do not call `read_league` per model.
- Test on a league that is in season. An out-of-season league's current
  pickle is empty. ncaafb is usually the right one.
- Report held-out and FBS-only numbers beside the pooled one, with a standard
  error. A single held-out ncaafb season swings by about 20e-4 Brier.

## Data jobs

- When a pull is costly to repeat, store nearly the whole response as a raw
  layer and derive tables from it.

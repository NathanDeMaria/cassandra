# Batch jobs

Runs cassandra's anchors/optimize/evaluate/publish pipeline on AWS Batch, on
the shared queue from [aws-batch-optimization][infra].

[infra]: https://github.com/NathanDeMaria/aws-batch-optimization

## What's here vs. what's shared

The split follows one line: **anything account-level is shared, anything that
moves when cassandra deploys lives here.**

| Shared (`aws-batch-optimization`) | Here (`cassandra/jobs`) |
| --- | --- |
| Job queue, compute environment, network | The five job definitions |
| ECR repo `cassandra` + its push user | Both schedules |
| `batch-execution-role` (pulls images) | `cassandra-batch-job-role` (what the code touches) |
| `batch-scheduler-role` (submits jobs) | |
| The `batch_job` and `job_schedule` modules | |

The deciding factor for keeping job definitions here is `image_tag`: a
definition pins a specific image, so it has to be applied by the same repo
whose CI pushed that image. Centralizing them would mean either a variable bump
in another repo on every push, or pinning `:latest` and giving up on
reproducible runs. Keeping them here also means a `terraform apply` for
cassandra can't destroy EndGame's schedules.

The shared modules are sourced over git at `?ref=main`. Terraform resolves
module sources before variables exist, so pinning them to a tag means editing
the `?ref=` in `main.tf` — there's deliberately no variable for it.

## The DAG

```
anchors   (array job, one child per league with division anchors — 3 today)
    |
    v
optimize  (array job, one child per league/model — 16 today)
    |
    +--> evaluate  (one job, scores everything, writes the metrics csv)
    |
    +--> publish   (array job, one child per league)
```

`anchors` runs first because it decides the scale everything downstream is on.
A team's anchor is the rating it starts at and regresses toward between
seasons, so a search run before the anchors exist is fit against a different
rating scale than the same search run after — and Brier score can't see the
difference, because it's dominated by games within a division. It's one node
ahead of the array rather than a step inside each optimize child because the
fit is per league: twenty children would fit the same three files twenty times,
and race each other writing them.

It's normally a no-op. `--if-missing` is on by default and checks the *bucket*,
not the container's disk, so once a league has anchors this is one s3 listing
and an exit. Refitting moves every rating the pipeline publishes, so it's
something you ask for — `jobs.py anchors --league ncaafb --no-if-missing`, or
deleting the object — not something the weekly run does to itself. `nfl` never
gets a child: 32 teams who all play each other have no tier gap to fit.

`evaluate` and `publish` are siblings, not a chain: `publish.py` reads
`<model>_result.json` and fits its own prob→margin mapping, so it needs the
optimizer's output but nothing evaluate produces.

**The edges are not in this terraform, and can't be.** Batch takes `dependsOn`
on `SubmitJob`, not on a job definition — so terraform declares the nodes and
`cassandra/batch/dag.py` declares the edges. That's what the fifth job
definition, `cassandra-launcher`, runs. It's a Batch job rather than a Lambda
so it runs the same image as the work it submits: the manifest it sizes the
array against has to be the one the children resolve their indices in, which is
only guaranteed if it's literally the same `models/` directory.

Both schedules target the launcher and differ only in command:

| Schedule | When | Command |
| --- | --- | --- |
| `cassandra-optimize-weekly` | Mon 03:00 CT | `submit` |
| `cassandra-publish-daily` | 09:00 CT | `submit --skip-optimize --skip-evaluate` |

Optimization is the expensive stage, so it's weekly. Publish is daily because
ratings move with new games every day even when the fitted parameters don't.
`--skip-optimize` implies skipping anchors: the anchors decide the scale a
*search* is fit against, and a republish reads that scale back out of s3
rather than deciding it.

## State between stages

The stages don't share a disk, so `~/.cassandra` goes through s3, mirrored
key-for-key under a `cassandra/` prefix in the batch bucket:

    ~/.cassandra/models/mens/elo_result.json
    s3://<batch-bucket>/cassandra/models/mens/elo_result.json

optimize uploads its result; evaluate and publish download the lot first. See
`cassandra/batch/artifacts.py`.

The anchors ride the same mirror, under `cassandra/predictor/data/`, and *every*
stage pulls them — not just the one that writes them. A result file carries the
fitted parameters but not the anchors (`PredictorConfig.params` is `float | str`,
and a per-team mapping is neither), so a replay rebuilds them by reading the
file. A publish container without it would ship ratings on a different scale
than the models were fit on, with nothing in the output saying so.

## Running it

```bash
# From the repo root -- these submit to Batch, they don't run locally.
make submit                                    # the whole DAG
make submit ARGS="--dry-run"                   # print what would be submitted, no AWS needed
make submit ARGS="--league mens"               # one league
make submit ARGS="--league mens --model elo --skip-evaluate"   # one model, as a test
make submit ARGS="--skip-optimize"             # re-evaluate and re-publish from s3
make submit ARGS="--skip-anchors"              # optimize, but don't re-check the anchors

poetry run python jobs.py manifest             # the work list, in array-index order
poetry run python jobs.py anchors --league ncaafb --no-if-missing   # force a refit
```

`./run_models.sh` still runs everything on one machine. It reads the same
manifest the array job does and the same `ANCHOR_LEAGUES` the anchors array is
sized against, so "optimize everything" means the same set of models, fit on
the same rating scale, locally and in the cloud.

## CI

`.github/workflows/terraform.yml` lints on every branch, plans on branches and
PRs, and applies on main -- the same shape as `aws-batch-optimization/infra`
and `endgame/jobs`.

| Job | When | Credentials |
| --- | --- | --- |
| `lint` | every push and PR touching `jobs/**` | none |
| `plan` | every branch and PR except main | `AWS_PLAN_ROLE_ARN` |
| `apply` | main only | `AWS_APPLY_ROLE_ARN` |

Two roles, not one: `terraform plan` executes provider code and runs on every
branch, so it must not be able to reach credentials that can apply. `oidc.tf`
creates both here rather than one repo minting roles for the others — the trust
policy is per-repository, so there's nothing to share and nothing to keep in
sync. The plan role gets `ReadOnlyAccess` plus write on this stack's state key
(a plan takes the lock, so it writes the `.tflock` beside the state even though
it changes nothing). The apply role gets `PowerUserAccess`, which denies IAM,
plus IAM scoped to `cassandra-*` — and `iam:PassRole` on the shared
`batch-execution-role` and `batch-scheduler-role`, which this stack references
but doesn't own: creating a job definition or a schedule passes them.

`lint` needs no credentials and no backend, so it runs before any of this
exists. Both other jobs check their role variable against `''` and skip while
it's unset, so the workflow lies dormant instead of failing red on every push
until you've wired the roles up:

```bash
cd jobs && make apply          # once, locally: creates the two CI roles
gh variable set AWS_PLAN_ROLE_ARN  --body "$(terraform output -raw ci_plan_role_arn)"
gh variable set AWS_APPLY_ROLE_ARN --body "$(terraform output -raw ci_apply_role_arn)"
```

Repository *variables*, not secrets — a role ARN isn't one, and the `!= ''`
check needs to be able to read it. The one secret this workflow reads is
`NOTIFICATION_EMAIL`, and leaving it unset is fine: an unset secret arrives as
`""` rather than as nothing, which `main.tf` normalises back to `null` so the
SNS topic and its rule simply aren't created.

`image_tag` stays at its `latest` default in CI, which is the tag a push to
main publishes. Pin a SHA in `terraform.tfvars` locally to make a run
reproducible.

## First-time setup

Ordering matters — steps 1–3 are in the other repo, and this project's
`terraform init` will fail until the shared modules exist on `main`.

1. **Shared infra.** In `aws-batch-optimization/infra`: `make apply`, then
   `make outputs`. That creates the `cassandra` ECR repo, its push user, and
   the shared roles, and writes `~/.aws-batch/config.json`.
2. **GitHub secrets** on this repo, for `.github/workflows/image.yml`:
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — from
     `terraform output -json ecr_iam_users`, the `cassandra` entry.
   - `IMAGE_URL` — from `terraform output -json repo_urls`.
   - `BATCH_CONFIG` — the contents of `~/.aws-batch/config.json`. It's baked
     into the image at `~dev/.aws-batch/config.json`; without it every job
     dies at startup looking for the bucket name.
3. **Push an image.** Merge to main, or `make push TAG=<sha>` locally.
4. **Here.** `cp terraform.tfvars.example terraform.tfvars`, set `image_tag`,
   then `make apply`.
5. **CI roles.** That apply created them; publish their ARNs as repository
   variables so the terraform workflow wakes up. See [CI](#ci).

`make update` re-fetches the shared modules — terraform caches git modules and
won't notice a change on the other end otherwise.

## Notes on the compute environment

The shared compute environment is all spot, all `.large` instances (2 vCPU,
8–16 GiB). Two consequences:

- **Nothing checkpoints.** A search reclaimed three hours in has produced
  nothing. The optimize definition retries 3 times, but only on host failure —
  a config that genuinely fails still fails once, rather than burning three
  copies of the same error.
- **Memory is the tight constraint**, not CPU: a whole league's seasons stay in
  memory for the length of a search. `optimize_memory` and `publish_memory` are
  variables for that reason. An `m6i.large` has 8192 MiB total but can't place
  a job asking for all of it — the ECS agent needs headroom.

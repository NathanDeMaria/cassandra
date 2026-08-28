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
| `image` (in `image.yml`) | build on PRs, push on main | `AWS_IMAGE_ROLE_ARN` |

Three roles, not one: `terraform plan` executes provider code and runs on every
branch, so it must not be able to reach credentials that can apply — and a
docker build has no business holding either. `oidc.tf`
creates both here rather than one repo minting roles for the others — the trust
policy is per-repository, so there's nothing to share and nothing to keep in
sync. The plan role gets `ReadOnlyAccess` plus write on this stack's state key
(a plan takes the lock, so it writes the `.tflock` beside the state even though
it changes nothing). The apply role gets `PowerUserAccess`, which denies IAM,
plus IAM scoped to `cassandra-*` — and `iam:PassRole` on the shared
`batch-execution-role` and `batch-scheduler-role`, which this stack references
but doesn't own: creating a job definition or a schedule passes them.

The image role is smaller than both: push/pull on the one ECR repository, the
account-wide `ecr:GetAuthorizationToken` that a docker login needs and that
takes no resource, and `s3:GetObject` on the shared stack's state object —
nothing else, and no write anywhere. It replaces the `ecr-pusher-cassandra`
user's long-lived access key, which used to live in this repo as two secrets.

`lint` needs no credentials and no backend, so it runs before any of this
exists. Every other job reads its role ARN through a `guard` step and skips
while it's unset, so the workflows lie dormant instead of failing red on every
push until you've wired the roles up:

```bash
cd jobs && make apply          # once, locally: creates the three CI roles
gh secret set AWS_PLAN_ROLE_ARN  --body "$(terraform output -raw ci_plan_role_arn)"
gh secret set AWS_APPLY_ROLE_ARN --body "$(terraform output -raw ci_apply_role_arn)"
gh secret set AWS_IMAGE_ROLE_ARN --body "$(terraform output -raw ci_image_role_arn)"
```

All three are repository *secrets*, and all three must be set the same way —
mixing the two contexts is what left the ECR repo empty once before, because
`vars.AWS_IMAGE_ROLE_ARN` reads as `''` when the ARN is stored as a secret and
the push then skips silently on a green run. That constraint is also why the
"is it wired up yet" check is a `guard` step rather than a job-level `if`: the
`secrets` context isn't available in a job `if`, and naming it there makes the
whole workflow invalid.

The other secret these workflows read is `NOTIFICATION_EMAIL`, and leaving it
unset is fine: an unset secret arrives as `""` rather than as nothing, which
`main.tf` normalises back to `null` so the SNS topic and its rule simply aren't
created.

`image_tag` stays at its `latest` default in CI, which is the tag a push to
main publishes. Pin a SHA in `terraform.tfvars` locally to make a run
reproducible.

### Where the image build gets its config

`config.json` — the bucket name for `Config.init_from_file`, the ECR URL for
the Makefile, the queue name for `jobs.py` — is **derived from the shared
stack's terraform state**, not pasted into a secret. The push job reads the
state object with the image role and filters it to those three keys:

```bash
aws s3 cp "s3://nathan-terraform/batch-state" - | jq '{bucket, repo_urls, job_queue_name} ...'
```

Two things that buys, beyond one less secret to rotate. It can't go stale: the
old `BATCH_CONFIG` secret was a snapshot of `terraform output -json` and had to
be re-pasted whenever the shared stack changed. And it stops shipping
credentials: `terraform output -json` includes sensitive values in full, so
that secret carried the `ecr_iam_users` access keys and baked them into every
image. The filter drops them. A key going missing fails the step rather than a
Batch job an hour later.

## Editing the terraform

`jobs/.devcontainer` is a second devcontainer, separate from the repo's Python
one: terraform and the AWS CLI, no poetry and no model dependencies. Two
containers rather than one because the two jobs share nothing — a terraform
edit doesn't want scikit-learn, and the Python container has no terraform, so
`make plan` there fails at the first command.

VS Code offers both when reopening in a container; this one is
**cassandra-infra**, and it opens in `jobs/` so terraform runs where the state
is. Its terraform version is pinned to the same one
`.github/workflows/terraform.yml` gives `hashicorp/setup-terraform` — a local
`make lint` that passes against a different terraform than CI runs is worth
very little, since `fmt` rules and `validate` diagnostics both move between
minor versions.

It runs as the same non-root `dev` user as the Python container, for the same
reason: `~/.aws` is bind-mounted from the host, and a root-owned SSO token
written inside the container is one the host can no longer refresh.

## First-time setup

Ordering matters — steps 1–3 are in the other repo, and this project's
`terraform init` will fail until the shared modules exist on `main`.

1. **Shared infra.** In `aws-batch-optimization/infra`: `make apply`, then
   `make outputs`. That creates the `cassandra` ECR repo, its push user, and
   the shared roles, and writes `~/.aws-batch/config.json`.
2. **Here.** `cp terraform.tfvars.example terraform.tfvars`, set `image_tag`,
   then `make apply`. Nothing needs a pushed image first — a Batch job
   definition naming a tag that doesn't exist yet applies fine.
3. **CI roles.** That apply created all three; publish their ARNs as
   repository variables so both workflows wake up. See [CI](#ci).
4. **Push an image.** Merge to main, or `make push TAG=<sha>` locally.

The only *secret* this repo needs is the optional `NOTIFICATION_EMAIL`. The
image workflow used to want four more; it reads the shared stack's terraform
state instead. See [Where the image build gets its config](#where-the-image-build-gets-its-config).

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

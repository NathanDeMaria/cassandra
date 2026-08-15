# Batch jobs

Runs cassandra's optimize/evaluate/publish pipeline on AWS Batch, on the shared
queue from [aws-batch-optimization][infra].

[infra]: https://github.com/NathanDeMaria/aws-batch-optimization

## What's here vs. what's shared

The split follows one line: **anything account-level is shared, anything that
moves when cassandra deploys lives here.**

| Shared (`aws-batch-optimization`) | Here (`cassandra/jobs`) |
| --- | --- |
| Job queue, compute environment, network | The four job definitions |
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
optimize  (array job, one child per league/model — 16 today)
    |
    +--> evaluate  (one job, scores everything, writes the metrics csv)
    |
    +--> publish   (array job, one child per league)
```

`evaluate` and `publish` are siblings, not a chain: `publish.py` reads
`<model>_result.json` and fits its own prob→margin mapping, so it needs the
optimizer's output but nothing evaluate produces.

**The edges are not in this terraform, and can't be.** Batch takes `dependsOn`
on `SubmitJob`, not on a job definition — so terraform declares the nodes and
`cassandra/batch/dag.py` declares the edges. That's what the fourth job
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

## State between stages

The stages don't share a disk, so `~/.cassandra` goes through s3, mirrored
key-for-key under a `cassandra/` prefix in the batch bucket:

    ~/.cassandra/models/mens/elo_result.json
    s3://<batch-bucket>/cassandra/models/mens/elo_result.json

optimize uploads its result; evaluate and publish download the lot first. See
`cassandra/batch/artifacts.py`.

## Running it

```bash
# From the repo root -- these submit to Batch, they don't run locally.
make submit                                    # the whole DAG
make submit ARGS="--dry-run"                   # print what would be submitted, no AWS needed
make submit ARGS="--league mens"               # one league
make submit ARGS="--league mens --model elo --skip-evaluate"   # one model, as a test
make submit ARGS="--skip-optimize"             # re-evaluate and re-publish from s3

poetry run python jobs.py manifest             # the work list, in array-index order
```

`./run_models.sh` still runs everything on one machine and is unchanged. It
reads the same manifest the array job does, so "optimize everything" means the
same set of models locally and in the cloud.

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

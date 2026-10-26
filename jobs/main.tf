provider "aws" {
  region = var.aws_region
}

# The shared account-level infrastructure: queue, compute environment, ECR
# repos, and the two roles that don't vary by app. Read rather than
# redeclared, so there is exactly one of each.
#
# aws-batch-optimization publishes its non-sensitive outputs as one JSON
# parameter (its infra/ssm.tf), in the same shape as a `terraform_remote_state`
# `outputs`, so this stack doesn't need to know where that one keeps its state.
data "aws_ssm_parameter" "shared" {
  name = var.shared_outputs_parameter
}

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

locals {
  shared = jsondecode(data.aws_ssm_parameter.shared.insecure_value)


  image = "${local.shared.repo_urls["cassandra"]}:${var.image_tag}"

  # The bucket cassandra reads seasons and odds from, and writes optimizer
  # results to under a `cassandra/` prefix.
  batch_bucket = local.shared.bucket

  # Where anything temporary goes -- search checkpoints, and any future
  # intermediate that only has to survive a retry. Provisioned and expired by
  # aws-batch-optimization; see `cassandra.constants.temp_bucket`.
  temp_bucket = local.shared.temp_bucket

  # What every job definition gets, whether or not it is known to need it.
  #
  # The region used to be the launcher's alone, on the reasoning that it was
  # the only stage calling an API that needs one to resolve an endpoint and
  # that the others only talk to s3, which botocore resolves without being
  # told. The first half is still true; the second half was only ever true of
  # botocore. It reaches instance metadata for a region when nothing else
  # supplies one, and `game_control` reads parquet through
  # `pyarrow.fs.S3FileSystem`, whose C++ SDK does not do that fallback. A
  # container has no `~/.aws/config` either, so with nothing in the
  # environment pyarrow resolved the empty region and every read came back
  # HTTP 301 naming the bucket's real one. That took the 2026-09-03 daily
  # publish down: `game_control` failed in 28 seconds and all six publish
  # children cascaded behind it.
  #
  # Set for every stage rather than for the one that is known to need it,
  # because "does this stage's s3 client happen to have a region fallback"
  # is a property of a library the stage imports, not of the stage.
  job_environment = [
    { name = "AWS_DEFAULT_REGION", value = var.aws_region },
    { name = "CASSANDRA_BUCKET", value = local.batch_bucket },
    { name = "CASSANDRA_TEMP_BUCKET", value = local.temp_bucket },
  ]

  # What the launcher submits. `game_control` used to be deliberately absent
  # -- its definition existed to be run by hand, and nothing in the DAG
  # pointed at it. Both sweeps are nodes again now that the blended models
  # read what they write, so both are here: a name in this map is a name the
  # launcher can start.
  job_definitions = {
    anchors        = module.anchors.name
    game_control   = module.game_control.name
    epa            = module.epa.name
    qb_out         = module.qb_out.name
    player_ratings = module.player_ratings.name
    optimize       = module.optimize.name
    evaluate       = module.evaluate.name
    publish        = module.publish.name
  }
}

# ------------------------------------------------------------------------------
# Job role: what cassandra's own code is allowed to touch
# ------------------------------------------------------------------------------
# Not the shared `job_role` from aws-batch-optimization, which covers the batch
# buckets and nothing else. Cassandra also writes releases to the webapp's
# artifacts bucket, and the launcher submits jobs -- both app-specific, so the
# role that grants them lives with the app.
data "aws_iam_policy_document" "job_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "job" {
  name               = "cassandra-batch-job-role"
  assume_role_policy = data.aws_iam_policy_document.job_assume.json
}

data "aws_iam_policy_document" "job" {
  statement {
    sid = "BatchBucketIO"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${local.batch_bucket}",
      "arn:aws:s3:::${local.batch_bucket}/*",
    ]
  }

  # The temp bucket: a search saves itself there under
  # `cassandra/checkpoints/<job id>.json` and deletes the save once it has
  # finished (`cassandra.checkpoint`). Delete is granted here and not on the
  # batch bucket: nothing a job writes there is its to remove.
  statement {
    sid = "TempBucketIO"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${local.temp_bucket}",
      "arn:aws:s3:::${local.temp_bucket}/*",
    ]
  }

  statement {
    sid = "PublishArtifacts"
    # Write-only on purpose: publish builds a release from scratch every run
    # and never reads back what it wrote. Rolling back is `cp` between keys,
    # done by hand.
    actions   = ["s3:PutObject"]
    resources = ["arn:aws:s3:::${var.artifacts_bucket}/*"]
  }

  # The picks job: email what it found, and read the Kalshi key it places
  # with at bedtime. Both scoped to the one topic and the one secret.
  statement {
    sid       = "EmailPicks"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.picks.arn]
  }

  statement {
    sid       = "ReadKalshiKey"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.kalshi.arn]
  }

  statement {
    sid = "SubmitOwnJobs"
    # The launcher runs as this role and submits the rest of the DAG. Batch
    # can't scope SubmitJob to "definitions this app owns" any finer than a
    # name prefix, and job definition ARNs carry a revision suffix, so this is
    # a wildcard over the account's definitions -- same as the shared
    # scheduler role.
    actions = [
      "batch:SubmitJob",
      "batch:DescribeJobs",
      "batch:ListJobs",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "job" {
  name   = "cassandra-job"
  role   = aws_iam_role.job.name
  policy = data.aws_iam_policy_document.job.json
}

# ------------------------------------------------------------------------------
# The DAG's nodes
# ------------------------------------------------------------------------------
# Nodes only. The edges -- anchors and the two sweeps feed optimize, which
# fans out, and evaluate and publish both wait on all of it -- are `dependsOn`
# arguments to SubmitJob, which a job definition has no field for.
# `cassandra/batch/dag.py` is where the DAG actually is.

# Fits the per-team regression anchors the search is scored against. Ahead of
# optimize in the DAG, and normally a no-op: `--if-missing` is on by default,
# so once a league has anchors in the bucket this is one s3 listing and an
# exit. The memory is `publish`'s rather than `optimize`'s because the fit
# reads every stored season for a league, the same as a release build does.
module "anchors" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-anchors"
  image              = local.image
  command            = ["anchors"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.publish_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

# Sweeps stored play-by-play into the per-game control index. Declared and
# unsubmitted for a while -- the `glicko_control` searches put its blend
# weight at zero in both leagues, so the models were deleted and no node
# pointed here (see `cassandra.predictor.control`). It is back in the DAG
# because `BlendedGlickoPredictor` and `BlendedMarginEloPredictor` read the
# index alongside the EPA one, which is the condition it was always going to
# earn a node back on.
#
# An array job now, one child per football league, where it used to be a
# single container doing both. The two are independent and each is an hour of
# s3 reads.
#
# Sized for what it holds rather than for how long it runs: a handful of
# NCAAFB weeks are decoded at once, ~20,000 plays each, plus the Arrow buffers
# they came out of and the league's seasons. It is idempotent on the win
# probability fit, so the common case re-reads one season rather than twenty.
module "game_control" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-game-control"
  image              = local.image
  command            = ["game_control"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.game_control_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

# The other sweep: the same weekly parquet, read for what each offense added
# per snap rather than for the shape of the win probability curve. Its own
# definition rather than a flag on `game_control` because the two are
# idempotent on different things -- control on the win probability fit, EPA on
# that plus the expected points fit -- so a retrain of one should re-sweep one.
#
# More memory than `game_control` for a concrete reason: that one reduces a
# game to a single float as it goes, while `epa_per_play` returns a `PlayEPA`
# per regulation snap, so a week's worth of NCAAFB is ~20,000 of those alive
# at once on top of the plays they came from.
module "epa" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-epa"
  image              = local.image
  command            = ["epa"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.epa_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

# The third sweep: the same weekly parquet, read for who took a snap at
# quarterback. Cheaper than the other two -- it is a regex over play text
# rather than a model over situations -- and it is a node because
# `models/{ncaafb,nfl}/glicko_full.json` and their siblings search
# `qb_out_penalty`, which is a dimension over an empty index until this has
# run.
module "qb_out" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-qb-out"
  image              = local.image
  command            = ["qb_out"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.qb_out_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

# The basketball sweep: no plays of its own to read, but the same place in the
# DAG. Each morning it runs ten-minute-version over the season's games so far
# and adds today's rows to `{league}_player_ratings.csv`, which
# `models/{mens,womens}/glicko_players.json` blend into their predictions --
# so publish waits on it the way it waits on the football sweeps. See
# `cassandra.player_ratings_build`.
module "player_ratings" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-player-ratings"
  image              = local.image
  command            = ["player_ratings"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.player_ratings_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

module "optimize" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-optimize"
  image              = local.image
  command            = ["optimize"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.optimize_memory
  timeout_seconds    = var.optimize_timeout_seconds

  # One vCPU rather than the module's two, so two searches share a box.
  #
  # When this was written every instance in the shared compute environment
  # was a 2-vCPU `.large` (it now also offers `.xlarge` and `.2xlarge`),
  # which made the vCPU the unit of cost here and memory nearly free: a
  # child reserving both cores owns the whole instance for its whole run
  # whatever it asks for in memory, and `optimize_memory` was reserving nine
  # times its measured peak for nothing.
  #
  # A search is single-threaded where it spends its time. One ncaafb
  # glicko_full probe measured 6.95s on two cores against 7.10s on one --
  # the replay is Python and a numba kernel, neither threaded. The one part
  # that reaches BLAS is bayes_opt's `suggest()`, which is 13% slower on a
  # single core, and on the ncaafb configs that is 9% of the search (24.8
  # minutes of suggest against ~4.2 hours of probes at glicko_compound's
  # 1000 iterations), so the second core buys ~1% of a child's wall clock.
  #
  # The exception, and the thing to watch: `suggest()` grows faster than
  # quadratically in observations, and nfl/glicko_full searches 2720 of
  # them, where a single suggest measured 171 seconds. That child is mostly
  # GP, not replay, so it pays closer to the full 13% -- and it is already
  # the child that spends 16.85 instance-hours over nine attempts and dies
  # on the 6h guard, 96 iterations short. Its problem is n_iter, not its
  # core count, and the fix is there rather than here.
  #
  # Against ~1%: two children per instance, so the stage's instance-hours
  # halve, and the same 16-vCPU cap runs 16 at once rather than 8. Optimize
  # is 49 of the weekly run's children and, measured over the 2026-09-28
  # run, 98% of its instance-hours.
  vcpu = 1

  # The compute environment is all spot. A search that gets reclaimed now
  # resumes from the checkpoint it keeps in the temp bucket (`cassandra.checkpoint`),
  # so a retry pays for the probes since the last save rather than for the
  # whole attempt. Only host failures retry; a config that genuinely fails
  # still fails once.
  #
  # Ten, which is as many as Batch allows. Three was measured to be too
  # close before the checkpoint existed: in the 20260903-230628 run, six of
  # twenty-four children were reclaimed and four of those spent all three
  # attempts. Six was then generous -- until 20260915-044450, when one
  # ncaafb search was reclaimed six times in 3.4 hours, five of them inside
  # fifty minutes, and failed with 597 saved probes it could have finished
  # from. With the checkpoint a retry costs at most the probes since the
  # last save, so the count is only ever a cap on how much spot churn a
  # search can outlast, and there is no reason to hold it under the limit.
  retry_attempts = 10

  # The same timeout, told to the search so it can stop short of it with a
  # result rather than be killed without one. See `optimize`'s `deadline`.
  #
  # And one BLAS thread, which is what makes `vcpu = 1` true. A vCPU here is
  # a CPU share, not a pin: the container still sees every core on the box,
  # and OpenBLAS starts a thread per core it sees. One child alone on a box
  # gets away with that, which is how the ~1% above was measured. Two
  # children with large GPs do not -- run 20261005-080205 lost evaluate and
  # every publish to ncaafb/glicko_full and nfl/glicko_margin timing out on
  # one m8i.large, a day after both finished in 3-5h. Resumed from that
  # run's saves (~800 and ~1000 observations) on one m8i.large, median
  # seconds a probe:
  #
  #                          ncaafb/glicko_full  nfl/glicko_margin
  #   alone                         4.6                6.1
  #   together                     52.9               46.3
  #   together, 1 thread each       7.0               10.2
  #
  # So capped, sharing a box costs 1.5-1.7x -- the two vCPUs of a .large are
  # one core's hyperthreads -- and uncapped it costs ten. All three names,
  # because which BLAS numpy links is a property of the wheel, not of this
  # repo.
  environment_variables = concat(local.job_environment, [
    { name = "CASSANDRA_ATTEMPT_TIMEOUT_SECONDS", value = tostring(var.optimize_timeout_seconds) },
    { name = "OMP_NUM_THREADS", value = "1" },
    { name = "OPENBLAS_NUM_THREADS", value = "1" },
    { name = "MKL_NUM_THREADS", value = "1" },
  ])
}

module "evaluate" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-evaluate"
  image              = local.image
  command            = ["evaluate"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.publish_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

module "publish" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-publish"
  image              = local.image
  command            = ["publish"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.publish_memory
  retry_attempts     = 3

  environment_variables = local.job_environment
}

# Prices the Kalshi ncaafb books against the model and emails the picks no
# earlier email had -- or, at bedtime, places them (`cassandra.picks_job`).
# Publish's memory, since it replays every stored season the way a release
# build does.
#
# One attempt, not three: a run that placed half its orders and then lost
# its host would place them again on a retry. A run that dies is picked up
# by the next hour's, which skips anything already held or resting.
module "picks" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-picks"
  image              = local.image
  command            = ["picks"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  memory             = var.publish_memory
  retry_attempts     = 1
  timeout_seconds    = 1800

  environment_variables = concat(local.job_environment, [
    { name = "CASSANDRA_PICKS_TOPIC_ARN", value = aws_sns_topic.picks.arn },
    { name = "CASSANDRA_KALSHI_SECRET", value = aws_secretsmanager_secret.kalshi.arn },
  ])
}

# The launcher: submits the other four with the right dependencies. It's a
# Batch job rather than a Lambda so it runs the same image as the work it
# submits -- the manifest it sizes the array against is the one the children
# will resolve indices in, which is only guaranteed if it's literally the same
# `models/` directory.
module "launcher" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/batch_job?ref=main"

  job_name           = "cassandra-launcher"
  image              = local.image
  command            = ["submit"]
  execution_role_arn = local.shared.batch_execution_role_arn
  job_role_arn       = aws_iam_role.job.arn
  # Submits and exits; it holds nothing in memory and waits on nothing.
  vcpu            = 1
  memory          = 1024
  timeout_seconds = 900

  # The region in `job_environment` matters most here: this is the stage that
  # calls Batch rather than s3, and until it was set both schedules died on
  # `NoRegionError` before submitting anything -- which reads as "no run
  # happened" rather than as a failure of the run. It is no longer the only
  # stage that needs one; see the local.
  environment_variables = concat(local.job_environment, [
    { name = "CASSANDRA_JOB_QUEUE", value = local.shared.job_queue_name },
    { name = "CASSANDRA_ANCHORS_JOB_DEFINITION", value = local.job_definitions.anchors },
    { name = "CASSANDRA_GAME_CONTROL_JOB_DEFINITION", value = local.job_definitions.game_control },
    { name = "CASSANDRA_EPA_JOB_DEFINITION", value = local.job_definitions.epa },
    { name = "CASSANDRA_QB_OUT_JOB_DEFINITION", value = local.job_definitions.qb_out },
    { name = "CASSANDRA_PLAYER_RATINGS_JOB_DEFINITION", value = local.job_definitions.player_ratings },
    { name = "CASSANDRA_OPTIMIZE_JOB_DEFINITION", value = local.job_definitions.optimize },
    { name = "CASSANDRA_EVALUATE_JOB_DEFINITION", value = local.job_definitions.evaluate },
    { name = "CASSANDRA_PUBLISH_JOB_DEFINITION", value = local.job_definitions.publish },
  ])
}

# ------------------------------------------------------------------------------
# Schedules
# ------------------------------------------------------------------------------
# Both target the launcher, differing only in command. Nothing else is
# scheduled: the stages are ordered by Batch dependencies, and a schedule
# can't express those.

module "weekly_run" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/job_schedule?ref=main"

  schedule_name       = "cassandra-optimize-weekly"
  schedule_expression = var.optimize_schedule
  schedule_timezone   = var.schedule_timezone
  job_definition      = module.launcher.name
  job_queue_arn       = local.shared.job_queue_arn
  scheduler_role_arn  = local.shared.batch_scheduler_role_arn
  command             = ["submit"]
}

module "daily_publish" {
  source = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/job_schedule?ref=main"

  schedule_name       = "cassandra-publish-daily"
  schedule_expression = var.publish_schedule
  schedule_timezone   = var.schedule_timezone
  job_definition      = module.launcher.name
  job_queue_arn       = local.shared.job_queue_arn
  scheduler_role_arn  = local.shared.batch_scheduler_role_arn
  # Republish from the results already in s3. Ratings move with new games
  # every day; the fitted parameters they're computed from don't.
  command = ["submit", "--skip-optimize", "--skip-evaluate"]
}

# ------------------------------------------------------------------------------
# Sweeps after finals
# ------------------------------------------------------------------------------
# EndGame starts its intraday football chain when games go final (its
# finals_watcher.tf), and the chain's last job writes the week's processed
# plays. The moment that job succeeds, this runs the two play-by-play sweeps a
# bet on the teams' next games reads: EPA, which carries the success rates,
# and who played quarterback. So a game that ended is in both indexes a few
# minutes later, instead of waiting for the next daily publish -- and a bet
# placed the night of a final is priced on everything its teams did in it.
#
# The contract between the two repos is EndGame's job name and nothing else:
# EndGame names no cassandra job, and nothing here reaches into its stack. A
# renamed chain step would stop these quietly, so the name is spelled out
# below rather than built from a pattern -- grep for it there.
#
# Both sweeps run in one job (`jobs.py sweeps`), scoped to the league whose
# plays changed: EPA re-sweeps the season in progress, then qb_out rebuilds
# its index whole, which takes a couple of minutes. One job because the picks
# run below has to start off the both of them, and EventBridge starts a
# target off one event -- as two jobs, the picks run started off the
# quarterbacks' and could beat the EPA upload. It runs EPA's definition (the
# bigger memory of the two), at its latest revision.
locals {
  # The job that writes a league's processed plays, in EndGame's intraday
  # chain (`intraday-process-football-plays-<league>`, plus the chain
  # module's `-chained-run`).
  football_plays_processed_jobs = {
    ncaafb = "intraday-process-football-plays-ncaafb-chained-run"
    nfl    = "intraday-process-football-plays-nfl-chained-run"
  }
}

resource "aws_cloudwatch_event_rule" "football_plays_processed" {
  for_each = local.football_plays_processed_jobs

  name        = "${var.resource_name_prefix}-football-plays-processed-${each.key}"
  description = "EndGame's intraday chain wrote ${each.key}'s processed plays"

  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail = {
      status  = ["SUCCEEDED"]
      jobName = [each.value]
    }
  })
}

resource "aws_cloudwatch_event_target" "sweeps_after_finals" {
  for_each = local.football_plays_processed_jobs

  rule      = aws_cloudwatch_event_rule.football_plays_processed[each.key].name
  target_id = "sweeps"
  arn       = local.shared.job_queue_arn
  role_arn  = aws_iam_role.sweeps_after_finals.arn

  batch_target {
    # By name, so a submission gets the latest revision -- what the launcher
    # submits too.
    job_definition = local.job_definitions.epa
    job_name       = "${var.resource_name_prefix}-sweeps-after-finals-${each.key}"
  }

  # EventBridge maps a `ContainerOverrides` key in a Batch target's input onto
  # SubmitJob's containerOverrides (command, environment, memory and vcpus
  # only).
  input = jsonencode({
    ContainerOverrides = {
      Command = ["sweeps", "--league", each.key]
    }
  })
}

# And once the finals sweeps have rebuilt both indexes, the picks job prices
# the books again. That's the run that sees a next-week matchup's second team
# finish: its game is in the season, its plays and quarterback are read, and
# the entry window has just opened. It skips its ESPN top-up (`--espn=False`):
# the chain wrote the season to s3 minutes ago.
resource "aws_cloudwatch_event_rule" "picks_after_finals" {
  name        = "${var.resource_name_prefix}-picks-after-finals"
  description = "The finals sweeps rebuilt ncaafb's EPA and quarterback indexes"

  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail = {
      status  = ["SUCCEEDED"]
      jobName = ["${var.resource_name_prefix}-sweeps-after-finals-ncaafb"]
    }
  })
}

resource "aws_cloudwatch_event_target" "picks_after_finals" {
  rule      = aws_cloudwatch_event_rule.picks_after_finals.name
  target_id = "picks"
  arn       = local.shared.job_queue_arn
  role_arn  = aws_iam_role.sweeps_after_finals.arn

  batch_target {
    job_definition = module.picks.name
    job_name       = "${var.resource_name_prefix}-picks-after-finals"
  }

  input = jsonencode({
    ContainerOverrides = {
      Command = ["picks", "--espn=False"]
    }
  })
}

resource "aws_iam_role" "sweeps_after_finals" {
  name = "${var.resource_name_prefix}-sweeps-after-finals"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = { Service = "events.amazonaws.com" }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sweeps_after_finals" {
  name = "${var.resource_name_prefix}-sweeps-after-finals"
  role = aws_iam_role.sweeps_after_finals.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "batch:SubmitJob"
        # The sweeps' definition and the picks job's, bare and at any
        # revision -- a submission by name resolves to the latest -- and the
        # queue they run on.
        Resource = concat(
          [local.shared.job_queue_arn],
          flatten([
            for name in [local.job_definitions.epa, module.picks.name] : [
              "arn:${data.aws_partition.current.partition}:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/${name}",
              "arn:${data.aws_partition.current.partition}:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/${name}:*",
            ]
          ]),
        )
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# Picks: when to look, where to email, and the key to place with
# ------------------------------------------------------------------------------
# Hourly through the windows a matchup's book goes from wide to priced: the
# backtest's entries after Saturday games tightened Sat 9pm-Sun 3am, and the
# edge was mostly gone by twelve hours (`picks.py --max-since-entry`). The
# finals rule above catches each entry as it opens; these catch the books
# that tighten after it. Weekday nights are for the midweek games.
module "picks_schedule" {
  source   = "git::https://github.com/NathanDeMaria/aws-batch-optimization.git//infra/modules/job_schedule?ref=main"
  for_each = var.picks_schedules

  schedule_name       = "${var.resource_name_prefix}-picks-${each.key}"
  schedule_expression = each.value
  schedule_timezone   = var.schedule_timezone
  job_definition      = module.picks.name
  job_queue_arn       = local.shared.job_queue_arn
  scheduler_role_arn  = local.shared.batch_scheduler_role_arn
  command             = ["picks"]
}

# Where the picks go. Subscribed by hand rather than here, so no address is
# in the repo -- an email subscription needs a click on its confirmation
# anyway, which terraform can't make:
#
#   aws sns subscribe --topic-arn "$(terraform output -raw picks_topic_arn)" \
#     --protocol email --notification-endpoint you@example.com
resource "aws_sns_topic" "picks" {
  name = "${var.resource_name_prefix}-picks"
}

# The Kalshi API key the job places with at bedtime, and reads the account
# with otherwise. Declared empty: the value is put by hand, so the key is
# never in terraform state --
#
#   aws secretsmanager put-secret-value --secret-id cassandra/kalshi-key \
#     --secret-string "$(jq -n --arg id "$(cat ~/.kalshi/key_id)" \
#       --rawfile pem ~/.kalshi/private_key.pem '{key_id: $id, private_key: $pem}')"
#
# Until it has a value the job emails picks sized on `--budget`, and never
# places.
resource "aws_secretsmanager_secret" "kalshi" {
  name        = "${var.resource_name_prefix}/kalshi-key"
  description = "Kalshi API key id and private key PEM, for cassandra-picks"
}

# ------------------------------------------------------------------------------
# Failure notification
# ------------------------------------------------------------------------------
# None here: aws-batch-optimization's alerts.tf emails on any job failing on
# the shared queue, these included (array children filtered, so a failed
# optimize array is one email). Its topic is `failure_topic_arn` in
# local.shared, for anything here that ever needs an alert a failed job
# wouldn't raise.

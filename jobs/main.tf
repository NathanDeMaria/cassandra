provider "aws" {
  region = var.aws_region
}

# The shared account-level infrastructure: queue, compute environment, ECR
# repos, and the two roles that don't vary by app. Read rather than
# redeclared, so there is exactly one of each.
data "terraform_remote_state" "shared" {
  backend = "s3"
  config = {
    bucket = var.shared_infra_state.bucket
    key    = var.shared_infra_state.key
    region = var.shared_infra_state.region
  }
}

data "aws_caller_identity" "current" {}

locals {
  shared = data.terraform_remote_state.shared.outputs

  image = "${local.shared.repo_urls["cassandra"]}:${var.image_tag}"

  # The bucket cassandra reads seasons and odds from, and writes optimizer
  # results to under a `cassandra/` prefix.
  batch_bucket = local.shared.bucket

  job_definitions = {
    anchors  = module.anchors.name
    optimize = module.optimize.name
    evaluate = module.evaluate.name
    publish  = module.publish.name
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

  statement {
    sid = "PublishArtifacts"
    # Write-only on purpose: publish builds a release from scratch every run
    # and never reads back what it wrote. Rolling back is `cp` between keys,
    # done by hand.
    actions   = ["s3:PutObject"]
    resources = ["arn:aws:s3:::${var.artifacts_bucket}/*"]
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
# Nodes only. The edges -- optimize fans out, evaluate and publish both wait on
# all of it -- are `dependsOn` arguments to SubmitJob, which a job definition
# has no field for. `cassandra/batch/submit.py` is where the DAG actually is.

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

  environment_variables = [
    { name = "CASSANDRA_BUCKET", value = local.batch_bucket },
  ]
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

  # The compute environment is all spot, and a search that gets reclaimed
  # three hours in has produced nothing -- there's no checkpointing to resume
  # from. Only host failures retry; a config that genuinely fails still fails
  # once.
  retry_attempts = 3

  environment_variables = [
    { name = "CASSANDRA_BUCKET", value = local.batch_bucket },
  ]
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

  environment_variables = [
    { name = "CASSANDRA_BUCKET", value = local.batch_bucket },
  ]
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

  environment_variables = [
    { name = "CASSANDRA_BUCKET", value = local.batch_bucket },
  ]
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

  environment_variables = [
    { name = "CASSANDRA_JOB_QUEUE", value = local.shared.job_queue_name },
    { name = "CASSANDRA_ANCHORS_JOB_DEFINITION", value = local.job_definitions.anchors },
    { name = "CASSANDRA_OPTIMIZE_JOB_DEFINITION", value = local.job_definitions.optimize },
    { name = "CASSANDRA_EVALUATE_JOB_DEFINITION", value = local.job_definitions.evaluate },
    { name = "CASSANDRA_PUBLISH_JOB_DEFINITION", value = local.job_definitions.publish },
  ]
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
# Failure notification
# ------------------------------------------------------------------------------
# A weekly job that quietly stops working is a model that quietly goes stale,
# and nothing else here would say so.
resource "aws_sns_topic" "failures" {
  count = var.notification_email == null ? 0 : 1
  name  = "cassandra-batch-failures"
}

resource "aws_sns_topic_subscription" "failures" {
  count     = var.notification_email == null ? 0 : 1
  topic_arn = aws_sns_topic.failures[0].arn
  protocol  = "email"
  endpoint  = var.notification_email
}

resource "aws_cloudwatch_event_rule" "job_failed" {
  count       = var.notification_email == null ? 0 : 1
  name        = "cassandra-batch-job-failed"
  description = "Any cassandra Batch job entering FAILED"

  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail = {
      status = ["FAILED"]
      # Array children report individually; without this a 20-child array
      # failing sends 20 emails and the parent's one is the useful one.
      jobDefinition = [{ prefix = "arn:aws:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/cassandra-" }]
    }
  })
}

resource "aws_cloudwatch_event_target" "job_failed" {
  count     = var.notification_email == null ? 0 : 1
  rule      = aws_cloudwatch_event_rule.job_failed[0].name
  target_id = "sns"
  arn       = aws_sns_topic.failures[0].arn
}

data "aws_iam_policy_document" "sns_publish" {
  count = var.notification_email == null ? 0 : 1

  statement {
    actions   = ["SNS:Publish"]
    resources = [aws_sns_topic.failures[0].arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_sns_topic_policy" "failures" {
  count  = var.notification_email == null ? 0 : 1
  arn    = aws_sns_topic.failures[0].arn
  policy = data.aws_iam_policy_document.sns_publish[0].json
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-2"
}

variable "image_tag" {
  description = "Image tag to run. CI tags every build with the short commit SHA; pin one here to make a run reproducible."
  type        = string
  default     = "latest"
}

variable "artifacts_bucket" {
  description = "Bucket the published ModelRelease artifacts go to, which the webapp serves from. Not the batch bucket."
  type        = string
  default     = "invisible-string-artifacts-080353813015"
}

variable "optimize_schedule" {
  description = "When to run the full DAG. Optimization is the expensive stage -- a config with n_iter in the hundreds replays every season once per probe -- so it runs weekly rather than nightly."
  type        = string
  default     = "cron(0 3 ? * MON *)"
}

variable "publish_schedule" {
  description = "When to re-publish without re-optimizing. Daily, so ratings pick up new games against the existing fitted parameters."
  type        = string
  default     = "cron(0 9 * * ? *)"
}

variable "schedule_timezone" {
  description = "Timezone both schedules are evaluated in"
  type        = string
  default     = "America/Chicago"
}

variable "optimize_timeout_seconds" {
  description = "Per-attempt wall clock for one model's search. The slowest checked-in config is a few hours; this is a runaway guard, not a target."
  type        = number
  default     = 21600
}

variable "optimize_memory" {
  description = "MiB for an optimize child. Every season for a league is held in memory for the whole search."
  type        = number
  default     = 4096
}

variable "publish_memory" {
  description = "MiB for a publish child. Higher than optimize: publish holds a league's seasons and the odds database at once."
  type        = number
  default     = 6144
}

variable "shared_infra_state" {
  description = "Where aws-batch-optimization keeps its state, read for the queue and the shared roles"
  type = object({
    bucket = string
    key    = string
    region = string
  })
  default = {
    bucket = "nathan-terraform"
    key    = "batch-state"
    region = "us-east-2"
  }
}

variable "notification_email" {
  description = "Where to email Batch failures. Null disables the topic, the rule and the subscription entirely."
  type        = string
  default     = null
}

# NB: there is deliberately no variable for the shared modules' git ref.
# Terraform requires `source` to be a literal -- it resolves modules before
# variables exist -- so pinning the shared modules means editing the `?ref=`
# in main.tf.

# ------------------------------------------------------------------------------
# CI
# ------------------------------------------------------------------------------

variable "github_repository" {
  description = "owner/repo allowed to assume the CI roles"
  type        = string
  default     = "NathanDeMaria/cassandra"
}

variable "github_owner_id" {
  description = <<-EOT
    Numeric ID of the GitHub account owning the repository.

    GitHub issues OIDC subjects in an immutable, ID-qualified form --
    `repo:OWNER@OWNER_ID/REPO@REPO_ID:...` -- rather than by name, so the trust
    policy has to match on IDs. Matching on names alone silently never matches
    and every assume fails with a generic "Not authorized".

      gh api users/NathanDeMaria --jq .id
  EOT
  type        = number
  default     = 5595197
}

variable "github_repository_id" {
  description = <<-EOT
    Numeric ID of the repository. See github_owner_id.

      gh api repos/NathanDeMaria/cassandra --jq .id
  EOT
  type        = number
  default     = 1184884301
}

variable "create_oidc_provider" {
  description = <<-EOT
    Create the GitHub Actions OIDC provider. Defaults false, like endgame and
    aws-batch-optimization: IAM permits exactly one provider per URL per
    account, and invisible-string creates the one in this account. If this
    account ever has none, set this true here and false there.
  EOT
  type        = bool
  default     = false
}

variable "state_bucket" {
  description = "Bucket holding terraform state. Plan needs write access for the lock file."
  type        = string
  default     = "nathan-terraform"
}

variable "state_key_prefix" {
  description = <<-EOT
    Key prefix within the state bucket that CI may lock and write. Matches the
    `key` in versions.tf; the trailing `*` in the policy covers the `.tflock`
    object `use_lockfile` writes beside it.
  EOT
  type        = string
  default     = "cassandra/jobs/terraform.tfstate"
}

variable "resource_name_prefix" {
  description = "Prefix for the IAM this stack creates. Scopes the apply role's IAM permissions."
  type        = string
  default     = "cassandra"
}

variable "ecr_repository_name" {
  description = <<-EOT
    The ECR repository the image workflow pushes to.

    Owned by the shared stack's `repos` module, not by this one -- named here
    only so the image role's policy can be scoped to it rather than to every
    repository in the account.
  EOT
  type        = string
  default     = "cassandra"
}

variable "shared_role_names" {
  description = <<-EOT
    Roles from the shared Batch stack that this one passes but does not manage.

    A Batch job definition is created with an execution role and an
    EventBridge schedule with a scheduler role, and creating either is an
    iam:PassRole on a role named by `aws-batch-optimization`, not by anything
    here. Listed by name rather than granting PassRole on `*`.
  EOT
  type        = list(string)
  default     = ["batch-execution-role", "batch-scheduler-role"]
}

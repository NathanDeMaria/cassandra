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

output "image" {
  description = "The image every job definition runs"
  value       = local.image
}

output "job_definitions" {
  description = "Names to pass to `jobs.py submit`, or to submit by hand"
  value       = local.job_definitions
}

output "launcher_job_definition" {
  description = "The definition the schedules submit"
  value       = module.launcher.name
}

output "job_role_arn" {
  value = aws_iam_role.job.arn
}

# ------------------------------------------------------------------------------
# CI
# ------------------------------------------------------------------------------
# Set these as repository *variables* (not secrets -- a role ARN isn't one, and
# the workflow checks them against '' to stay dormant until they're wired up):
#   gh variable set AWS_PLAN_ROLE_ARN  --body "$(terraform output -raw ci_plan_role_arn)"
#   gh variable set AWS_APPLY_ROLE_ARN --body "$(terraform output -raw ci_apply_role_arn)"

output "ci_plan_role_arn" {
  description = "role-to-assume for plan jobs (any branch, any PR)"
  value       = aws_iam_role.ci_plan.arn
}

output "ci_apply_role_arn" {
  description = "role-to-assume for apply jobs (main only)"
  value       = aws_iam_role.ci_apply.arn
}

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

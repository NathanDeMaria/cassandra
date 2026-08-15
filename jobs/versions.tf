terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }

  # Cassandra's own state, separate from the shared infrastructure it reads.
  # That separation is the point of keeping this in the app repo: applying a
  # new image tag here can't touch the queue, the compute environment, or
  # another repo's schedules.
  backend "s3" {
    bucket       = "nathan-terraform"
    key          = "cassandra/jobs/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}

TF_ARGS=-var-file=terraform.tfvars

_init:
	terraform init

plan: _init
	terraform plan $(TF_ARGS)

apply: _init
	terraform apply $(TF_ARGS)

# The shared modules are sourced from git, so terraform caches them and will
# not notice a change on the other end until told to look again.
update:
	terraform init -upgrade

# What CI runs on every branch. Needs no credentials and no backend, so it
# works anywhere -- including before the OIDC roles are wired up, where it is
# the only signal the terraform workflow gives.
lint:
	terraform fmt -check -recursive
	terraform init -backend=false -input=false
	terraform validate

.PHONY: _init plan apply update lint

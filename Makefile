lint:
	poetry run ruff check --fix .
	poetry run ty check .


# Same checks as `lint`, but reports instead of fixing (what CI runs)
check:
	poetry run ruff check .
	poetry run ty check .


test:
	poetry run pytest .


# Condense the newest Batch run into a report. Pass a run id to pick an older
# one, e.g. `make report ARGS=20260829-022910`; `--list` shows what's there.
# Needs credentials -- AWS_PROFILE, or `--cached` to re-read the last fetch.
report:
	poetry run python .claude/skills/run-report/summarize_run.py $(ARGS)


# Build a release for every model in every league, locally. Reads the seasons
# and odds once for the whole run, so it's minutes rather than the half hour a
# process per model would spend re-reading s3.
publish:
	poetry run python publish.py --upload


# ------------------------------------------------------------------------------
# Batch
# ------------------------------------------------------------------------------
# The terraform output from aws-batch-optimization (`make outputs` there). A
# local copy wins so CI can drop one in from a secret without a home directory.
CONFIG := $(firstword $(wildcard config.json $(HOME)/.aws-batch/config.json))
IMAGE_URL ?= $(shell jq -r .repo_urls.value.cassandra $(CONFIG))
# Lazy, not `:=`. An immediate assignment shells out to STS on every make
# invocation, including `make build`, which needs no credentials at all --
# and in CI that means an error on stderr before a build that then succeeds.
ACCOUNT = $(shell aws sts get-caller-identity --query "Account" --output text)
REGION ?= us-east-2
TAG ?= local

IS_MAIN := $(shell git rev-parse --abbrev-ref HEAD | grep -q ^main$$ && echo true || echo false)

# CI passes buildx cache flags in here; empty locally, where the daemon's own
# layer cache already does the job.
CACHE_FLAGS ?=

# `build` and `push` differ only in their output flag, so they stay one build
# definition -- tagging included, rather than a follow-up `docker tag`.
BUILD_FLAGS := --target runtime -f .devcontainer/Dockerfile -t ${IMAGE_URL}:${TAG}
ifeq ($(IS_MAIN),true)
BUILD_FLAGS += -t ${IMAGE_URL}:latest
endif

# What `build` does with the result. `--load` locally, where the point is to
# have the image; empty in CI, where the point is only that it compiles and
# loading it would spend a minute unpacking layers nothing reads.
BUILD_OUTPUT ?= --load

build:
	docker buildx build ${CACHE_FLAGS} ${BUILD_FLAGS} ${BUILD_OUTPUT} .

_ecr_login:
	aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# `--push` uploads straight from the builder, skipping the `--load` tarball
# round trip whose only purpose would be giving `docker push` something to
# read. Needs the ECR login both for the push and for the registry cache CI
# passes in CACHE_FLAGS.
push: _ecr_login
	docker buildx build ${CACHE_FLAGS} ${BUILD_FLAGS} --push .


# Submit the whole DAG: optimize (one array child per model) then evaluate and
# publish. Pass through anything jobs.py takes, e.g.
#   make submit ARGS="--league mens --dry-run"
ARGS ?=
submit:
	poetry run python jobs.py submit $(ARGS)


.PHONY: lint check test report publish build push _ecr_login submit

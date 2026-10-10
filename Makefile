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


# The four below read a model's replay through `cassandra.replay_cache`: the
# first run of a model replays it (minutes, about what one `evaluate` child
# costs) and keeps it under ~/.cassandra/replays; later runs read it back in
# a second, until the config, the replay's code, the lock file or a day
# changes. Pass `--refresh` in ARGS to replay anyway.

# Slice one model's residuals to see where it's wrong, e.g.
# `make diagnose ARGS="--league nfl --model margin_blend"`.
diagnose:
	poetry run python diagnose.py $(ARGS)


# The team-seasons a model keeps missing, shrunk by how much of that is
# noise, with the market's view beside each, e.g. `make team-seasons
# ARGS="--league ncaafb --model glicko_margin_units --season 2026"`. Add
# `--team "Indiana Hoosiers"` for one team's game log, `--by conference` for
# conference-seasons.
team-seasons:
	poetry run python team_seasons.py $(ARGS)


# Test an idea -- a table of team-season or game features -- against a
# model's residuals: effect, null, market, and a cross-validated MAE/brier
# change, e.g. `make evidence ARGS="--league ncaafb --model glicko_full
# --team-seasons ideas.csv --division FBS --since 2015"`.
evidence:
	poetry run python evidence.py $(ARGS)


# Grade a model's picks the way a bettor would: whether it knows anything
# the line doesn't, closing line value from the first line after each team's
# previous game, spread strategies with p-values, and moneyline strategies.
# Several models at once compare, e.g. `make betting ARGS="--league ncaafb
# --model glicko_full glicko_margin_units"`. The odds history is kept too,
# for three hours.
betting:
	poetry run python betting.py $(ARGS)


# What to bet on Kalshi this week: the model against the live order books.
# Run it once the night's last game is final -- the edge `betting` found is
# at that first price and fades within a day. `--offline` when the AWS login
# has lapsed, e.g. `make picks ARGS="--offline --stake 25"`.
picks:
	poetry run python picks.py $(ARGS)


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

# Whether this build also moves `latest`. On main by default, which is what
# `latest` means -- but CI's per-architecture builds turn it off, because
# `latest` has to end up on the manifest list rather than on whichever
# architecture happened to push last. See the workflow.
TAG_LATEST ?= $(IS_MAIN)

# Baked into the image, where an optimize child reads it to decide whether
# the code behind a fit has moved (`cassandra.fingerprint`). Lazy, so it only
# shells out for a target that builds. A dirty tree still reports the commit
# it sits on, which would let a search skip against uncommitted changes --
# the reason to care is a local `make push`, so it prints a warning there
# rather than silently lying.
GIT_SHA ?= $(shell git rev-parse HEAD)

# Which architectures to build for. Both are deployable: the compute
# environment can launch Graviton instances as well as x86 ones, and the
# instance type is Batch's choice at scale-up time, not CI's -- so what ECR
# holds has to cover either. One platform at a time here rather than a list,
# because a list is a manifest list, and that is something only a registry can
# hold: `--load` imports into the local docker daemon, which cannot. CI builds
# each architecture on a runner of that architecture and merges the two
# afterwards; a laptop pushing by hand wants `PLATFORM=` and gets its own.
PLATFORM ?=

# `build` and `push` differ only in their output flag, so they stay one build
# definition -- tagging included, rather than a follow-up `docker tag`.
BUILD_FLAGS := --target runtime -f .devcontainer/Dockerfile -t ${IMAGE_URL}:${TAG}
BUILD_FLAGS += --build-arg GIT_SHA=$(GIT_SHA)
ifeq ($(TAG_LATEST),true)
BUILD_FLAGS += -t ${IMAGE_URL}:latest
endif
ifneq ($(PLATFORM),)
BUILD_FLAGS += --platform $(PLATFORM)
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
	@git diff --quiet HEAD || echo "WARNING: uncommitted changes -- this image will claim it was built from $(GIT_SHA), and a search may skip against code that isn't in that commit"
	docker buildx build ${CACHE_FLAGS} ${BUILD_FLAGS} --push .

# Join the two per-architecture images into the manifest list that the commit
# tag and `latest` actually point at. A registry-side operation: it reads the
# two manifests and writes an index referring to them, so nothing is rebuilt
# or re-uploaded.
#
# The `-amd64`/`-arm64` tags it reads exist only because `--push` has to put
# each half somewhere a later step can name. Nothing pulls them, and the
# repository's lifecycle policy expires them with every other commit tag.
#
#   make manifest TAG=sha-abc1234
manifest: _ecr_login
	docker buildx imagetools create \
		-t ${IMAGE_URL}:${TAG} \
		$(if $(filter true,$(TAG_LATEST)),-t ${IMAGE_URL}:latest) \
		${IMAGE_URL}:${TAG}-amd64 ${IMAGE_URL}:${TAG}-arm64


# Submit the whole DAG: optimize (one array child per model) then evaluate and
# publish. Pass through anything jobs.py takes, e.g.
#   make submit ARGS="--league mens --dry-run"
ARGS ?=
submit:
	poetry run python jobs.py submit $(ARGS)


.PHONY: lint check test report diagnose team-seasons evidence betting picks publish build push manifest _ecr_login submit

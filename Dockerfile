# What runs on Batch: `python jobs.py <command>`, and nothing else. The
# devcontainer is .devcontainer/Dockerfile, and is the template's.
#
# 3.14-bookworm.
#
# The digest of the *index*, not of one platform's manifest inside it. A
# single-platform digest has nothing for `--platform linux/arm64` to resolve
# to, so the build either ignores the request or fails outright. This is the
# one line that makes a multi-architecture build possible at all.
#
#   docker buildx imagetools inspect python:3.14-bookworm
#
# The full image rather than -slim because it already has git, which uv needs to
# clone the git dependencies (endgame, call-it-what-you-want, ...) at their
# pinned revs -- so there is no apt step.
FROM python@sha256:b3c121f5b6b446c964c6ea924d9a099e259b29d7b56df82729e33572a31eadcc

# uv at the version mirrorball pins, copied out of its own image (by index
# digest, ghcr.io/astral-sh/uv:0.13.6) rather than pip-installed into the
# interpreter the job runs on.
COPY --from=ghcr.io/astral-sh/uv@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /uvx /usr/local/bin/

# Non-root, uid/gid 1000, as it has been: one user across every stage and the
# devcontainer means ~/.cassandra and ~/.aws-batch/config.json resolve to the
# same paths in a job as on a laptop.
RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash dev

USER dev

# ~/.aws-batch is where config.json lands below; ~/.endgame is where a job
# writes its game cache. Created here so they are owned by dev.
RUN mkdir -p /home/dev/.aws-batch /home/dev/.endgame
WORKDIR /home/dev/runtime

# The interpreter above, never one uv downloads; bytecode compiled at build
# time so a job doesn't pay for it on every start. The venv is first on PATH,
# so the entrypoint's bare `python` is the one the packages were installed into.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/home/dev/runtime/.venv/bin:${PATH}

# Dependencies before the code, so editing it doesn't reinstall them.
#
# `--no-dev --group fit`: `fit` carries scikit-learn, bayes_opt, numba, fire and
# endgame_aws -- optimize and publish are both fitting paths, and both read the
# bucket, so the deployed image needs the group the webapp deliberately doesn't.
# See the comment on the group in pyproject.toml. (`default-groups` there
# includes `dev`, so the flag that drops it is what keeps the tooling out.)
#
# numba is the one of those that changes what a job *costs* rather than what
# it can do: without it every Glicko search still runs, three times slower
# per probe, and the 6h `optimize_timeout_seconds` is what notices. There is
# no compiler in this image and none needed -- it installs from a wheel.
#
# `uv sync` fails if the lock file and pyproject.toml disagree (--locked) and
# if a group is missing, so unlike the poetry install this replaced, a spelling
# that quietly installs less is an error.
COPY --chown=dev:dev pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group fit --no-install-project

# `Config.init_from_file` looks here second, and it's the only one of its two
# candidates worth using: the first is the *package* root, which for an
# installed endgame_aws is site-packages. Required at build time, so a missing
# bucket is a build failure rather than every job failing at import. CI writes
# it from the shared stack's outputs first; see .github/workflows/image.yml.
COPY --chown=dev:dev config.json /home/dev/.aws-batch/config.json

# models/ has to come along: the manifest is derived from the checked-in
# configs at runtime, so the image is what defines the work list.
ADD --chown=dev:dev . /home/dev/runtime

# What this image was built from, which is how an optimize child decides
# whether the code behind a fit has changed (`cassandra.fingerprint`). The
# whole commit, deliberately: working out which modules feed one search is
# the kind of cleverness whose failure mode is serving a stale fit forever.
#
# Last, and after the ADD, because a build arg invalidates every layer below
# it and this one changes on every commit -- the ADD already does, so putting
# it here costs nothing and putting it higher would cost the install.
#
# Empty by default. `cassandra.fingerprint.code_version` reads an empty value
# as "unknown", which never matches a recorded fingerprint, so an image built
# without this searches everything rather than skipping on a false match.
ARG GIT_SHA=""
ENV CASSANDRA_GIT_SHA=$GIT_SHA

ENTRYPOINT ["python", "/home/dev/runtime/jobs.py"]
CMD ["manifest"]

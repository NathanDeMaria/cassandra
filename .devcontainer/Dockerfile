# 3.14-bookworm
FROM python@sha256:b58de395e525ffb2dbc6804ed76d46f053885aa8ed0f3ba5df3da5be12e7bf9b AS base

# Non-root, uid/gid 1000 to match the host user whose files get bind-mounted in
# (~/.aws, ~/.claude, ~/.cassandra). Running as root here meant every SSO
# token refresh clobbered the host's cache with root-owned files. The Batch
# stage inherits the user rather than switching back: it has nothing to do as
# root, and one user across every stage means ~/.cassandra and
# ~/.aws-batch/config.json resolve to the same paths in a job as on a laptop.
#
# The chown is for the runtime stage, which installs with no virtualenv and so
# treats /usr/local as the environment. Three paths under it have to be
# writable by dev:
#   site-packages  installed packages
#   bin            console scripts
#   src            git checkouts. Poetry clones VCS deps (endgame_aws) into
#                  `env.path / "src"`, NOT into the cache dir -- so with no
#                  virtualenv that resolves to /usr/local/src, and a root-owned
#                  one fails the install with EPERM.
# The rest of the stdlib stays root-owned; chowning it would copy 53M into
# this layer for nothing. Here rather than in `runtime` so the chown walks an
# empty site-packages: after the install it would copy every installed package
# into the layer.
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev \
    && mkdir -p /usr/local/src \
    && chown -R dev:dev /usr/local/lib/python3.14/site-packages /usr/local/bin /usr/local/src

USER dev
WORKDIR /home/dev

# Parents of the bind mounts in devcontainer.json. Docker creates a missing
# mount point as root, and chowning on the host can't fix these because the
# parents themselves aren't mounted -- so `aws configure` and anything else
# writing a new file straight into ~/.aws would hit EPERM. Pre-creating them
# as dev means docker finds them already there and leaves ownership alone.
# ~/.aws-batch is also where the runtime stage lands its config.json.
RUN mkdir -p /home/dev/.aws /home/dev/.aws-batch /home/dev/.endgame

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1
ENV PATH=/home/dev/.local/bin:${PATH}

COPY --chown=dev:dev poetry.* ./
COPY --chown=dev:dev pyproject.toml ./

# Deliberately no `poetry install` here. The two stages below want opposite
# things from poetry -- runtime needs the packages where a bare `python` finds
# them, the devcontainer wants the in-project venv poetry.toml asks for -- and
# a shared install layer can only be one of those. Each stage installs for
# itself; nobody builds both in one go, so the "shared" layer was never shared.


# What actually gets deployed to Batch. Kept free of the dev tooling below so
# the pushed image stays small and CI never builds the interactive extras.
FROM base AS runtime

# No venv, unlike the devcontainer: the entrypoint is `python`, not
# `poetry run`, so the packages have to be somewhere the interpreter looks
# without being activated. An env var rather than editing poetry.toml, which
# is free to keep asking for the venv a laptop wants.
ENV POETRY_VIRTUALENVS_CREATE=false

# `fit` carries scikit-learn, bayes_opt and fire -- optimize and publish are
# both fitting paths, so the deployed image needs the group the webapp
# deliberately doesn't. See the comment on the group in pyproject.toml.
#
# `--only main,fit`, NOT `--only main --with fit`: poetry ignores `--with`
# whenever `--only` is given, and says so in a line that scrolls past in a
# green build. That spelling installed neither scikit-learn nor
# bayesian-optimization, and the image looked fine -- `fire` comes in
# transitively through endgame, so the one name you'd think to check was
# there. Every optimize job would have died importing bayes_opt.
RUN poetry install --no-root --only main,fit

# `Config.init_from_file` looks here second, and it's the only one of its two
# candidates worth using: the first is the *package* root, which for an
# installed endgame_aws is site-packages. Required at build time, so a missing
# bucket is a build failure rather than every job failing at import. CI writes
# it from a secret first; see .github/workflows/image.yml.
COPY --chown=dev:dev config.json /home/dev/.aws-batch/config.json

# models/ has to come along: the manifest is derived from the checked-in
# configs at runtime, so the image is what defines the work list.
ADD --chown=dev:dev . /home/dev/runtime
WORKDIR /home/dev/runtime

ENTRYPOINT ["python", "/home/dev/runtime/jobs.py"]
CMD ["manifest"]


# The devcontainer target. Branches off `base` so editing source doesn't
# invalidate the awscli layer on a local rebuild.
FROM base AS dev

# The two installers that write outside $HOME. `sudo` rather than a `USER root`
# / `USER dev` sandwich so the layer's files land owned by whoever should own
# them without a follow-up chown.
#
# GitHub CLI, for PR and issue work from a Claude session. Debian ships no
# usable gh package and the upstream apt repo would be an unpinned external
# source, so this takes the release .deb at a pinned version instead. dpkg
# rather than apt-get because the package lists are deleted above; gh's only
# dependency is git, which the base image already has.
ARG GH_VERSION=2.97.0
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL -o /tmp/gh.deb "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${arch}.deb" \
    && sudo dpkg -i /tmp/gh.deb \
    && rm /tmp/gh.deb

# Only needed interactively (`aws sso login`, `make push`) -- nothing in
# cassandra shells out to the CLI, it goes through aiobotocore.
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" &&\
    unzip awscliv2.zip &&\
    sudo ./aws/install &&\
    rm awscliv2.zip &&\
    rm -r ./aws

# jq, for the Makefile's IMAGE_URL: `make build` and `make push` read the ECR
# repo url out of ~/.aws-batch/config.json with it, and without jq make prints
# "jq: No such file or directory" on *every* invocation -- including targets
# like `make test` that never look at IMAGE_URL, because that line is a
# `$(shell ...)` make expands up front.
#
# apt-get, unlike the gh install above, because jq is in Debian and pinning a
# release binary by hand buys nothing here. `base` deletes the package lists,
# so this refreshes them and drops them again rather than leaving the index in
# the layer. Interactive-only, like the awscli: the runtime stage builds
# images, it doesn't build them.
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends jq \
    && sudo rm -rf /var/lib/apt/lists/*

# No `poetry install` here either, and this one is worth spelling out:
# poetry.toml asks for an in-project venv, and "in project" at build time
# means /home/dev/.venv -- but VS Code opens the bind-mounted workspace at
# /workspaces/cassandra, so that venv would be built into the image and then
# never used. devcontainer.json's postCreateCommand runs the install in the
# workspace instead, where the .venv it makes is the one the editor and
# `poetry run` both pick up.

# Claude Code, pinned like poetry above. Installs the native build to
# ~/.local/share/claude/versions/<v> and symlinks a launcher into
# ~/.local/bin, which the PATH above already covers. The installer's scratch
# dir (~/.claude/downloads) is emptied before the layer closes, so nothing
# here collides with the ~/.claude bind mount at runtime. Auto-updates still
# run inside a live container; this only pins what a fresh build starts from.
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.235

# Claude's top-level config lives at $HOME/.claude.json, which is *outside* the
# ~/.claude bind mount -- so a rebuild loses it and the first `claude` run
# replays the onboarding prompts. Seeding these two keys skips that; Claude
# fills in the rest of the file on first run.
RUN printf '%s' '{"hasCompletedOnboarding":true,"theme":"dark"}' \
    > /home/dev/.claude.json

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/home/dev/.bashrc"

# Machine-local environment -- the AWS_PROFILE to use, mainly. It has to be
# settable without committing it, so it lives in .devcontainer/local.env,
# which .gitignore and .dockerignore both drop; devcontainer.json passes its
# path down as DEVCONTAINER_ENV_FILE (from ${containerWorkspaceFolder}, since
# where the workspace lands depends on which folder you opened).
#
# Sourced from two places on purpose, because bash reads different files
# depending on how it was started and both cases matter here:
#   .bashrc    interactive shells -- a VS Code terminal
#   BASH_ENV   non-interactive ones -- `bash -c`, which is how Claude Code and
#              anything scripted runs commands
# Between them every bash in the container ends up with the profile set,
# rather than just the ones a human typed into.
#
# Keep this script silent: BASH_ENV is read before *every* non-interactive
# bash, so anything it prints lands in the output of every `$(...)` capture.
COPY --chown=dev:dev .devcontainer/local-env.sh /home/dev/.local-env.sh
ENV BASH_ENV=/home/dev/.local-env.sh
RUN echo '. /home/dev/.local-env.sh' >> "/home/dev/.bashrc"

# Is this a good way to add stuff on load?
# Not sure, but it's working
COPY --chown=dev:dev profile.sh /home/dev/profile.sh
RUN echo ". profile.sh" >> "/home/dev/.bashrc"

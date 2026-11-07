# Template-owned (mirrorball): the devcontainer only. A repo that ships an
# image keeps its runtime Dockerfile elsewhere and owns it.
#
# 3.15.2-trixie on 2026-10-28. Pinned by the digest of the multi-platform index, so
# the same line builds on amd64 and arm64.
FROM python@sha256:a51b0cd81c9c0fc08afc4a823843b459dc71c9386a02ccb150097b41d783f33b

# uv at a pinned version, copied out of its own image rather than pip-installed,
# so it shares nothing with the project's environment. Bump it on purpose: uv is
# what reads uv.lock.
COPY --from=ghcr.io/astral-sh/uv@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /uvx /usr/local/bin/

# The interpreter is this image's, not one uv downloads to satisfy
# requires-python.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3
# The workspace is a bind mount, so uv's cache and .venv are on different
# filesystems and hardlinking between them fails.
ENV UV_LINK_MODE=copy

# Non-root, uid/gid 1000 to match the host user whose files get bind-mounted in
# (~/.aws, ~/.claude). As root, every token refresh would leave root-owned files
# in the host's directories.
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev

USER dev
WORKDIR /home/dev
ENV PATH=/home/dev/.local/bin:${PATH}

# No `uv sync` here: the .venv belongs in the bind-mounted workspace, which
# doesn't exist at build time. devcontainer.json's postCreateCommand does it.

# The GitHub CLI, from the release .deb at a pinned version (Debian's is too
# old, and the upstream apt repo is unpinned).
ARG GH_VERSION=2.97.0
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL -o /tmp/gh.deb "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${arch}.deb" \
    && sudo dpkg -i /tmp/gh.deb \
    && rm /tmp/gh.deb

# For `aws sso login` and poking at buckets by hand. ~/.aws is pre-created so
# the bind mount's parent is dev's, not root's.
RUN mkdir -p /home/dev/.aws \
    && arch="$(uname -m)" \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o awscliv2.zip \
    && unzip -q awscliv2.zip \
    && sudo ./aws/install \
    && rm -r awscliv2.zip ./aws

# Claude Code, pinned. Auto-updates still run in a live container; this only
# pins what a fresh build starts from.
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.235

# $HOME/.claude.json is outside the ~/.claude bind mount, so a rebuild loses it
# and replays onboarding. Seeding these two keys skips that.
RUN printf '%s' '{"hasCompletedOnboarding":true,"theme":"dark"}' \
    > /home/dev/.claude.json

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/home/dev/.bashrc"

# Machine-local environment (AWS_PROFILE, mainly) from the untracked
# .devcontainer/local.env. Sourced by interactive shells (.bashrc) and by
# non-interactive ones (BASH_ENV), which is how Claude Code runs commands. Keep
# it silent: anything it prints lands in every `$(...)` capture.
COPY --chown=dev:dev .devcontainer/local-env.sh /home/dev/.local-env.sh
ENV BASH_ENV=/home/dev/.local-env.sh
RUN echo '. /home/dev/.local-env.sh' >> "/home/dev/.bashrc"

COPY --chown=dev:dev .devcontainer/profile.sh /home/dev/profile.sh
RUN echo ". profile.sh" >> "/home/dev/.bashrc"

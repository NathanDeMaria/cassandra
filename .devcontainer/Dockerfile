# 3.14-bookworm
FROM python@sha256:b58de395e525ffb2dbc6804ed76d46f053885aa8ed0f3ba5df3da5be12e7bf9b

# Non-root, uid/gid 1000 to match the host user whose files get bind-mounted in
# (~/.aws, ~/.claude, ~/.cassandra). Running as root here meant every SSO
# token refresh clobbered the host's cache with root-owned files.
#
# poetry.toml sets virtualenvs.create = false, so poetry treats /usr/local as
# the environment and needs three paths under it writable by dev:
#   site-packages  installed packages
#   bin            console scripts
#   src            git checkouts. Poetry clones VCS deps (endgame_aws) into
#                  `env.path / "src"`, NOT into the cache dir -- so with no
#                  virtualenv that resolves to /usr/local/src, and a root-owned
#                  one fails the install with EPERM.
# The rest of the stdlib stays root-owned; chowning it would copy 53M into
# this layer for nothing.
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
RUN mkdir -p /home/dev/.aws /home/dev/.aws-batch /home/dev/.endgame

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/home/dev/.bashrc"

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1
ENV PATH /home/dev/.local/bin:${PATH}

# Is this a good way to add stuff on load?
# Not sure, but it's working
COPY --chown=dev:dev profile.sh /home/dev/profile.sh
RUN echo ". profile.sh" >> "/home/dev/.bashrc"

COPY --chown=dev:dev poetry.* ./
COPY --chown=dev:dev pyproject.toml ./
RUN poetry install --no-root

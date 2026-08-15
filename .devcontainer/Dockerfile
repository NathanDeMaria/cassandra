# 3.14-bookworm
FROM python@sha256:b58de395e525ffb2dbc6804ed76d46f053885aa8ed0f3ba5df3da5be12e7bf9b AS base

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1
ENV PATH /root/.local/bin:${PATH}

COPY poetry.* ./
COPY pyproject.toml ./
# `fit` carries scikit-learn, bayes_opt and fire -- optimize and publish are
# both fitting paths, so the deployed image needs the group the webapp
# deliberately doesn't. See the comment on the group in pyproject.toml.
RUN poetry install --no-root --only main --with fit


# What actually gets deployed to Batch. Kept free of the dev tooling below so
# the pushed image stays small and CI never builds the interactive extras.
FROM base AS runtime

# `Config.init_from_file` looks here second, and it's the only one of its two
# candidates worth using: the first is the *package* root, which for an
# installed endgame_aws is site-packages. Required at build time, so a missing
# bucket is a build failure rather than every job failing at import. CI writes
# it from a secret first; see .github/workflows/on_push.yml.
COPY config.json /root/.aws-batch/config.json

# models/ has to come along: the manifest is derived from the checked-in
# configs at runtime, so the image is what defines the work list.
ADD . /root/runtime
WORKDIR /root/runtime

ENTRYPOINT ["python", "/root/runtime/jobs.py"]
CMD ["manifest"]


# The devcontainer target. Branches off `base` so editing source doesn't
# invalidate the awscli layer on a local rebuild.
FROM base AS dev

RUN poetry install --no-root --with fit

# Only needed interactively (`aws sso login`, `make push`) -- nothing in
# cassandra shells out to the CLI, it goes through aiobotocore.
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" &&\
    unzip awscliv2.zip &&\
    ./aws/install &&\
    rm awscliv2.zip &&\
    rm -r ./aws

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

# Is this a good way to add stuff on load?
# Not sure, but it's working
COPY profile.sh /root/profile.sh
RUN echo ". profile.sh" >> "/root/.bashrc"

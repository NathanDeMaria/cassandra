# 3.14-bookworm
FROM python@sha256:b58de395e525ffb2dbc6804ed76d46f053885aa8ed0f3ba5df3da5be12e7bf9b

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1
ENV PATH /root/.local/bin:${PATH}

# Is this a good way to add stuff on load?
# Not sure, but it's working
COPY profile.sh /root/profile.sh
RUN echo ". profile.sh" >> "/root/.bashrc"

COPY poetry.* ./
COPY pyproject.toml ./
RUN poetry install --no-root

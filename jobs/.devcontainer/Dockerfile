# :22.04 -- same pinned base as aws-batch-optimization/infra/.devcontainer, so
# the two infra containers don't drift apart for no reason.
FROM ubuntu@sha256:104ae83764a5119017b8e8d6218fa0832b09df65aae7d5a6de29a85d813da2fb

ENV DEBIAN_FRONTEND=noninteractive

# Pinned, and pinned to the same version `.github/workflows/terraform.yml`
# gives `hashicorp/setup-terraform`. A local `make lint` that passes against a
# different terraform than CI runs is worth very little -- `terraform fmt`
# rules and validate diagnostics both move between minor versions.
ARG TERRAFORM_VERSION=1.15.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    jq \
    less \
    openssh-client \
    sudo \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip" \
    -o /tmp/terraform.zip \
    && unzip -q /tmp/terraform.zip -d /usr/local/bin \
    && rm /tmp/terraform.zip \
    && terraform version

# Non-root, uid/gid 1000, for the same reason the repo's other devcontainer is:
# ~/.aws is bind-mounted from the host, and a root-owned SSO token written
# inside the container is one the host then can't refresh. The upstream
# infra container runs as root and predates that fix; this one doesn't inherit
# the problem just to match.
RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev

USER dev
WORKDIR /home/dev

# Parents of the bind mounts in devcontainer.json. Docker creates a missing
# mount point as root, so pre-creating them as dev is what keeps `aws sso
# login` and `make outputs` able to write there.
RUN mkdir -p /home/dev/.aws /home/dev/.aws-batch

# `terraform` talks to AWS through the SDK and needs no CLI, but refreshing an
# expired SSO token does, and `make outputs` shells out to it.
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && sudo /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo "$SNIPPET" >> /home/dev/.bashrc

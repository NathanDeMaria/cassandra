# Sourced by every bash in the devcontainer -- from .bashrc when interactive,
# from BASH_ENV when not. See the block that installs it in ./Dockerfile.
#
# Keep this silent and cheap. BASH_ENV means it runs ahead of every
# non-interactive bash, so anything printed here ends up inside the output of
# every `$(...)` in every script that runs in this container.

# The machine-local settings themselves (AWS_PROFILE and anything else that
# shouldn't be committed) live in .devcontainer/local.env, which is gitignored.
# devcontainer.json passes its path down as DEVCONTAINER_ENV_FILE; the file
# itself is optional, and absent it everything just behaves as it did before.
if [ -f "${DEVCONTAINER_ENV_FILE:-}" ]; then
    . "${DEVCONTAINER_ENV_FILE}"
fi

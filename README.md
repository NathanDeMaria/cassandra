# cassandra

Team ratings and win probabilities

## Development

Open the folder in its devcontainer, or with [uv](https://docs.astral.sh/uv/)
installed:

```
make check   # ruff format --check, ruff check, ty -- what CI runs
make lint    # same, fixing what it can
make test    # pytest
```

Tooling comes from the [mirrorball](https://github.com/NathanDeMaria/mirrorball)
template; `.copier-answers.yml` records which version.

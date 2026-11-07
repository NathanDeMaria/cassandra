# Template-owned (mirrorball): `copier update` changes this file. Targets of
# this repo's own go in project.mk, which is included at the bottom.
#
# `uv run` syncs the environment from uv.lock before it runs anything, so these
# work from a fresh checkout with no install step in front of them.

# Fix what can be fixed, then report what's left
lint:
	uv run ruff check --fix .
	uv run ruff format .
	uv run ty check .


# Same checks as `lint`, but reports instead of fixing (what CI runs)
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check .


test:
	uv run pytest .


.PHONY: lint check test

-include project.mk

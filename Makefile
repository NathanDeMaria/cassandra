lint:
	poetry run ruff check --fix .
	poetry run ty check .


# Same checks as `lint`, but reports instead of fixing (what CI runs)
check:
	poetry run ruff check .
	poetry run ty check .


test:
	poetry run pytest .


# Kick off a full optimize+eval run in the background. $$ escapes the dollar so
# make passes it through to the shell instead of expanding it itself.
run-all:
	@log="logs/$$(date +%Y-%m-%d_%H-%M-%S).log"; \
		nohup ./run_models.sh > "$$log" 2>&1 & \
		printf 'running in background\ntail -f %s\nto follow\n' "$$log"


# Build a release for every model in every league, locally. Reads the seasons
# and odds once for the whole run, so it's minutes rather than the half hour a
# process per model would spend re-reading s3.
publish:
	poetry run python publish.py --upload


.PHONY: lint check test run-all publish

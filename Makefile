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


.PHONY: lint check test run-all

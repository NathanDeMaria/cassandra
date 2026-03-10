lint:
	poetry run ruff check --fix .
	poetry run mypy .


test:
	poetry run pytest .

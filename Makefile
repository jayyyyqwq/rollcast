.PHONY: build setup test vet lint fmt

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

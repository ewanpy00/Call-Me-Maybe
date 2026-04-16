f ?= data/input/functions_definition.json
i ?= data/input/function_calling_tests.json
o ?= data/output/result.json

install:
	uv sync

run:
	uv run python -m src -function_definition $(f) -input $(i) -output $(o)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf src/__pycache__
	rm -rf data/output/*
	rm -rf llm_sdk/llm_sdk/__pycache__
	rm -rf data/output

lint:
	flake8 --exclude .venv,llm_sdk
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

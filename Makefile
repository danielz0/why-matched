.PHONY: install test test-fast lint bench

install:
	pip install -e ".[local,viz,dev]"

test:
	pytest -q

test-fast:
	pytest -q -m "not optional_deps"

lint:
	ruff check .
	ruff format --check .

bench:
	python benchmarks/negation_benchmark.py $(ARGS)

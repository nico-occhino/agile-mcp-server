.PHONY: install test eval schema run

install:
	pip install -e ".[dev,eval]"

test:
	pytest tests/

eval:
	python -m scripts.eval

schema:
	python -m scripts.export_schema

run:
	python -m client.main

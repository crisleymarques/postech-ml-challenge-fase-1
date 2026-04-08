.PHONY: install lint format test clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e .

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache
	rm -rf .pytest_cache
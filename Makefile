.PHONY: install lint format test quality train-baselines mlflow-ui clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest tests/

quality: lint test

train-baselines:
	$(PYTHON) -m src.train_baselines

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache
	rm -rf .pytest_cache
.PHONY: install lint format test quality train-baselines train-mlp mlflow-ui clean

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

train-mlp:
	$(PYTHON) -m src.train_mlp --epochs 100 --patience 10 --batch-size 32

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri "sqlite:///$(CURDIR)/mlflow.db"

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache
	rm -rf .pytest_cache
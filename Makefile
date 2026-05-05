.PHONY: install lint format test quality train-baselines train-mlp mlflow-ui clean

PYTHON ?= python3
<<<<<<< HEAD
=======
MLFLOW_TRACKING_URI ?= sqlite:///$(abspath mlflow.db)
>>>>>>> f8290b60dbc7640f2f4cc0fb4d3618e51dad89f7

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
<<<<<<< HEAD
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns
=======
	$(PYTHON) -m mlflow ui --backend-store-uri "$(MLFLOW_TRACKING_URI)"
>>>>>>> f8290b60dbc7640f2f4cc0fb4d3618e51dad89f7

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache
	rm -rf .pytest_cache
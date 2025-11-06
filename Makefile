.PHONY: help prepare-merged split-multilabel train-lstm train-bert eval-multilabel api demo test test-cov lint install

help: ## Show this help message
	@echo "Toxicity Detector - Available Commands:"
	@echo ""
	@echo "Data Preparation:"
	@echo "  prepare-merged     Prepare merged dataset from raw data"
	@echo "  split-multilabel   Split dataset into train/validation/test sets"
	@echo ""
	@echo "Model Training:"
	@echo "  train-lstm         Train LSTM model for multilabel classification"
	@echo "  train-bert         Train BERT model for multilabel classification"
	@echo ""
	@echo "Evaluation & Deployment:"
	@echo "  eval-multilabel    Evaluate trained models and generate metrics"
	@echo "  api                Start FastAPI server for model inference"
	@echo "  demo               Launch Streamlit demo application"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test               Run pytest tests"
	@echo "  test-cov           Run pytest tests with coverage report"
	@echo "  lint               Run flake8 linting"
	@echo ""
	@echo "Setup:"
	@echo "  install            Install package in editable mode"
	@echo ""
	@echo "Usage: make <target>"
	@echo "Example: make train-bert"

prepare-merged:
	python src/prepare_merged.py

split-multilabel:
	python src/split_multilabel.py

train-lstm:
	python src/train_lstm.py

train-bert:
	python src/train_bert.py

eval-multilabel:
	python src/evaluate_multilabel.py

api:
	uvicorn src.api:app --reload --port 8000

demo:
	streamlit run app/streamlit_app_multilabel.py

test: ## Run pytest tests
	pytest tests/ -v

test-cov: ## Run pytest tests with coverage report
	pytest tests/ -v --cov=src --cov=app --cov-report=html --cov-report=term

lint: ## Run flake8 linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

install: ## Install package in editable mode
	pip install -e .

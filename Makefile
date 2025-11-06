.PHONY: help prepare-merged split-multilabel train-lstm train-bert eval-multilabel api demo test test-cov lint install package-release package-assets update-release

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
	@echo "Release Management:"
	@echo "  package-release    Create single-ZIP release package (toxicity-detector-v<version>.zip)"
	@echo "  package-assets     [Legacy] Create zip archive of models and processed data only"
	@echo "  update-release     [Legacy] Update existing GitHub release with new assets"
	@echo ""
	@echo "Usage: make <target>"
	@echo "Example: make train-bert"

prepare-merged:
	python3 src/prepare_merged.py

split-multilabel:
	python3 src/split_multilabel.py

train-lstm:
	python3 src/train_lstm.py

train-bert:
	python3 src/train_bert.py

eval-multilabel:
	python3 src/evaluate_multilabel.py

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

# Release Management
package-release: ## Create single-ZIP release package (requires VERSION variable)
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make package-release VERSION=1.0.1"; \
		exit 1; \
	fi
	python3 tools/pack_release.py --version $(VERSION)
	@echo "Release package created: dist/toxicity-detector-v$(VERSION).zip"

# Legacy release management (kept for backward compatibility)
ASSET_ZIP := toxicity-detector-assets-$(shell date +%Y%m%d).zip

package-assets: ## [Legacy] Create zip archive of models and processed data only
	@rm -f $(ASSET_ZIP)
	@zip -r $(ASSET_ZIP) \
		models/*.pth models/*config.json models/*_test_preds.npy models/*_test_labels.npy \
		data/processed/train_multilabel.csv data/processed/val_multilabel.csv data/processed/test_multilabel.csv
	@echo "Created $(ASSET_ZIP)"

update-release: ## [Legacy] Update existing GitHub release with new assets
	@if [ -z "$(TAG)" ]; then \
		echo "Error: TAG is required. Usage: make update-release TAG=v1.0.0 FILES='...' BODY='...'"; \
		exit 1; \
	fi
	@bash scripts/release_update.sh $(TAG) "$(FILES)" "$(BODY)"

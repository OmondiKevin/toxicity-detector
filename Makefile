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

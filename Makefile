prepare:
	python src/prepare_multiclass.py

train-mc:
	python src/train_multiclass_baseline.py

eval-mc:
	python src/evaluate_multiclass.py

api:
	uvicorn src.api:app --reload --port 8000

demo:
	streamlit run app/streamlit_app.py
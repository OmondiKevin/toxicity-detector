from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import os

APP_NAME = "toxicity-detector-api"
MODEL_PATH = os.getenv("MODEL_PATH", "models/multiclass_tfidf_svm.joblib")

app = FastAPI(title=APP_NAME, version="0.1.0")

# Load model once at startup
bundle = joblib.load(MODEL_PATH)
PIPE = bundle["pipeline"]
LABEL_MAP = bundle.get("label_map", {1: "hate", 2: "offensive", 3: "neutral"})


class MCIn(BaseModel):
    texts: List[str]


class MCOut(BaseModel):
    predictions: List[int]
    labels: List[str]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, Any]:
    return {"app": APP_NAME, "version": app.version, "model_path": MODEL_PATH}


@app.post("/classify_multiclass", response_model=MCOut)
def classify_multiclass(inp: MCIn) -> MCOut:
    preds = PIPE.predict(inp.texts)
    labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]
    return MCOut(predictions=[int(p) for p in preds], labels=labels)

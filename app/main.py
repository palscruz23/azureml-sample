from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib

app = FastAPI()

model = None
feature_names = None

@app.on_event("startup")
def load_model():
    global model, feature_names
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "outputs" / "model" / "model.joblib"
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(payload: dict):
    if model is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    if "input_data" not in payload:
        raise HTTPException(status_code=400, detail="Request must include 'input_data'")
    rows = payload["input_data"]

    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="'input_data' must be a list")

    matrix = []

    for row in rows:
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail="Each item in 'input_data' must be an object")

        missing_features = [name for name in feature_names if name not in row]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        ordered_row = [row[name] for name in feature_names]
        matrix.append(ordered_row)

    predictions = model.predict(matrix)
    return {"predictions": predictions.tolist()}



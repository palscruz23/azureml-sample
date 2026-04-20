# Azure ML Sample

This project is a learning exercise for deploying a simple machine learning solution to Azure ML.

Current focus:
- train a basic regression model with scikit-learn
- serve predictions through a custom FastAPI API
- prepare the project for Azure ML deployment later

## Current Status

Implemented so far:
- regression training script in [src/train.py](/home/palscruz23/azureml-sample/src/train.py)
- FastAPI inference app in [app/main.py](/home/palscruz23/azureml-sample/app/main.py)
- model artifact saved to `outputs/model/model.joblib`
- local endpoint routes:
  - `GET /health`
  - `POST /score`

Not done yet:
- Dockerfile for the FastAPI service
- Azure ML deployment configuration
- endpoint deployment and cloud validation

## Project Flow

1. `src/train.py` trains a `LinearRegression` model using the scikit-learn diabetes dataset.
2. The script saves a model bundle with:
   - the trained model
   - the feature names used during training
3. `app/main.py` loads that model bundle on startup.
4. The FastAPI app exposes `/score` for prediction requests.

## Run Locally

Install dependencies with your preferred workflow. This repo currently declares dependencies in `pyproject.toml`.

Train the model:

```bash
python3 src/train.py
```

This should create:

```text
outputs/model/model.joblib
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Send a Prediction Request

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": [
      {
        "age": 0.038075906,
        "sex": 0.05068012,
        "bmi": 0.061696207,
        "bp": 0.021872354,
        "s1": -0.044223498,
        "s2": -0.034820763,
        "s3": -0.043400846,
        "s4": -0.002592262,
        "s5": 0.019907486,
        "s6": -0.017646125
      }
    ]
  }'
```

Expected response shape:

```json
{
  "predictions": [178.4]
}
```

The exact prediction value may differ slightly depending on the trained model artifact.

## API Contract

`POST /score` expects JSON in this shape:

```json
{
  "input_data": [
    {
      "age": 0.0,
      "sex": 0.0,
      "bmi": 0.0,
      "bp": 0.0,
      "s1": 0.0,
      "s2": 0.0,
      "s3": 0.0,
      "s4": 0.0,
      "s5": 0.0,
      "s6": 0.0
    }
  ]
}
```

Validation currently checks:
- `input_data` must exist
- `input_data` must be a list
- each item must be an object
- each item must include all expected features

## Next Steps For Azure ML

To turn this into an Azure ML deployment with a custom API, the next work items are:

1. Add a `Dockerfile` to package the FastAPI application.
2. Add a production startup command for the API server.
3. Create Azure ML managed online endpoint configuration.
4. Deploy the container to Azure ML as a custom inference service.
5. Test the deployed endpoint with the same JSON payload used locally.

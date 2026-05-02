# Azure ML Sample

This project is a learning exercise for deploying a simple machine learning solution to Azure ML.

Current focus:
- train a basic regression model with scikit-learn
- serve predictions through a custom FastAPI API
- package the API as a Docker image
- push the image to Azure Container Registry
- deploy the image to an Azure ML managed online endpoint

## Current Status

Implemented so far:
- regression training script in [src/train.py](/home/palscruz23/azureml-sample/src/train.py)
- FastAPI inference app in [app/main.py](/home/palscruz23/azureml-sample/app/main.py)
- model artifact saved to `outputs/model/model.joblib`
- Dockerfile for the FastAPI service
- Azure ML endpoint configuration in `endpoint.yml`
- Azure ML deployment configuration in `deployment.yml`
- Docker image pushed to Azure Container Registry:
  - `diabetes20260422.azurecr.io/azureml-sample:v1`
- Azure ML online endpoint created:
  - `diabetes-endpoint-20260422`
- local endpoint routes:
  - `GET /health`
  - `POST /score`

Not done yet:
- Azure ML online deployment and scoring validation

## Project Flow

1. `src/train.py` trains a `LinearRegression` model using the scikit-learn diabetes dataset.
2. The script saves a model bundle with:
   - the trained model
   - the feature names used during training
3. `app/main.py` loads that model bundle on startup.
4. The FastAPI app exposes `/score` for prediction requests.
5. The `Dockerfile` packages the FastAPI app and model artifact into a container image.
6. The image is tagged for Azure Container Registry as `diabetes20260422.azurecr.io/azureml-sample:v1`.
7. The image is pushed to Azure Container Registry.
8. The Azure ML online endpoint is created from `endpoint.yml`.
9. Azure ML will use the ACR image when creating the online deployment from `deployment.yml`.

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

## Docker Image

The Docker image is built from the project root.

```bash
docker build \
  -t diabetes20260422.azurecr.io/azureml-sample:v1 \
  .
```

Command breakdown:

- `docker build` creates an image from the `Dockerfile`.
- `-t` assigns the image name and tag.
- `diabetes20260422.azurecr.io` is the Azure Container Registry login server.
- `azureml-sample` is the repository/image name inside ACR.
- `v1` is the image version tag.
- `.` means the current project folder is the Docker build context.

Verify the local image:

```bash
docker images
```

Expected repository and tag:

```text
diabetes20260422.azurecr.io/azureml-sample   v1
```

## Push Image To ACR

Log in to Azure Container Registry:

```bash
az acr login --name diabetes20260422
```

Push the image:

```bash
docker push diabetes20260422.azurecr.io/azureml-sample:v1
```

Docker knows to push to ACR because the image name starts with the registry server:

```text
diabetes20260422.azurecr.io
```

Verify the image is in ACR:

```bash
az acr repository list \
  --name diabetes20260422 \
  -o table
```

Check the image tag:

```bash
az acr repository show-tags \
  --name diabetes20260422 \
  --repository azureml-sample \
  -o table
```

Expected tag:

```text
v1
```

## Next Steps For Azure ML

The image is now available in ACR and the online endpoint has been created. The next work items are:

1. Create the Azure ML online deployment from `deployment.yml`.
2. Confirm traffic is routed to the `blue` deployment.
3. Test the deployed endpoint with the same JSON payload used locally.

Create the endpoint:

```bash
az ml online-endpoint create \
  --file endpoint.yml \
  --resource-group poljohncruz-rg \
  --workspace-name my-workspace
```

If endpoint creation fails with `SubscriptionNotRegistered` and the missing provider is shown as `[N/A]`, check these providers:

```bash
az provider show --namespace Microsoft.PolicyInsights --query registrationState -o tsv
az provider show --namespace Microsoft.Cdn --query registrationState -o tsv
```

If either one is not registered, register it:

```bash
az provider register --namespace Microsoft.PolicyInsights
az provider register --namespace Microsoft.Cdn
```

Wait until both return:

```text
Registered
```

Then retry the endpoint creation command.

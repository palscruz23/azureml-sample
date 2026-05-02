# Azure ML YAML Guide

This guide explains how to create the two Azure ML YAML files used by this project:

- `endpoint.yml`
- `deployment.yml`

The short version:

```text
endpoint.yml   = public API front door
deployment.yml = running container behind that front door
```

## endpoint.yml

`endpoint.yml` creates the Azure ML online endpoint. The endpoint is the stable API entry point that clients call.

This project uses:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: diabetes-endpoint-20260422
auth_mode: key
```

Field breakdown:

- `$schema` tells tools which Azure ML schema to use for validation and editor hints.
- `name` is the Azure ML online endpoint name.
- `auth_mode: key` means callers need an endpoint key to send requests.

Create the endpoint with:

```bash
az ml online-endpoint create \
  --file endpoint.yml \
  --resource-group poljohncruz-rg \
  --workspace-name my-workspace
```

Check the endpoint with:

```bash
az ml online-endpoint show \
  --name diabetes-endpoint-20260422 \
  --resource-group poljohncruz-rg \
  --workspace-name my-workspace \
  -o table
```

## deployment.yml

`deployment.yml` creates the actual running service behind the endpoint. This is where Azure ML starts the Docker container.

This project uses:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: blue
endpoint_name: diabetes-endpoint-20260422

environment:
  image: diabetes20260422.azurecr.io/azureml-sample:v1
  inference_config:
    liveness_route:
      port: 80
      path: /health
    readiness_route:
      port: 80
      path: /health
    scoring_route:
      port: 80
      path: /score

instance_type: Standard_DS2_v2
instance_count: 1
```

Field breakdown:

- `name: blue` names this deployment. `blue` is a common name for the first stable deployment.
- `endpoint_name` connects this deployment to the endpoint from `endpoint.yml`.
- `environment.image` is the ACR image Azure ML should pull and run.
- `liveness_route` tells Azure ML how to check whether the container process is alive.
- `readiness_route` tells Azure ML how to check whether the container is ready for traffic.
- `scoring_route` tells Azure ML where to send prediction requests.
- `instance_type` chooses the VM size for the deployment.
- `instance_count` chooses how many container instances to run.

Create the deployment with:

```bash
az ml online-deployment create \
  --file deployment.yml \
  --resource-group poljohncruz-rg \
  --workspace-name my-workspace \
  --all-traffic
```

`--all-traffic` sends 100% of endpoint requests to this deployment.

## How The Files Connect

The endpoint name and deployment `endpoint_name` must match:

```yaml
# endpoint.yml
name: diabetes-endpoint-20260422
```

```yaml
# deployment.yml
endpoint_name: diabetes-endpoint-20260422
```

The image in `deployment.yml` must match the image pushed to ACR:

```yaml
image: diabetes20260422.azurecr.io/azureml-sample:v1
```

The route paths in `deployment.yml` must match the FastAPI routes in `app/main.py`:

```yaml
path: /health
path: /score
```

## Request Flow

After both files are applied, the flow is:

```text
client request
  -> Azure ML online endpoint
  -> blue deployment
  -> Docker container from ACR
  -> FastAPI /score route
  -> model prediction response
```

## Common Mistakes

- The endpoint exists, but no deployment exists yet.
- The deployment image name does not match the pushed ACR image.
- The endpoint identity does not have `AcrPull` permission on ACR.
- The route paths in `deployment.yml` do not match the routes exposed by the app.
- The VM size is larger than the available Azure quota.

#!/bin/bash
set -e

# Source the environment variables from the root-based path
source "gcp_assets/config/env.sh"

echo "Deploying services to project: ${PROJECT_ID} in region: ${REGION}"

# Build container images using Cloud Build
echo "Building container images using Cloud Build for project ${PROJECT_ID}..."
gcloud builds submit \
  --config "cloudbuild.yaml" \
  --substitutions=_PROJECT_ID="${PROJECT_ID}",\
_API_SERVICE_NAME="${API_SERVICE_NAME}"
echo "Build completed."


# Deploy API service to Cloud Run
echo "Deploying Backend-API service to CloudRun..."
OPENAI_API_KEY=$(gcloud secrets versions access latest --secret="${OPENAI_SECRET_NAME}")

echo "Deploying API service to Cloud Run..."
gcloud run deploy "${API_SERVICE_NAME}" \
    --image "${API_IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --set-env-vars "QDRANT_HOST=${QDRANT_HOST}, OPENAI_API_KEY=${OPENAI_API_KEY}" \
    --service-account "${API_SA}" \
    --port 8080
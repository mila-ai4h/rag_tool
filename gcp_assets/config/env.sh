#!/bin/bash
# Global environment variables

# GCP project & location
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ZONE="us-central1-a"

# Service names
export API_SERVICE_NAME="backend-api"

# Container image
export API_IMAGE="gcr.io/${PROJECT_ID}/${API_SERVICE_NAME}:latest"

# Service accounts
export API_SA_NAME="backend-api-sa"
export API_SA="${API_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Secrets
export OPENAI_SECRET_NAME="openai-api-key"

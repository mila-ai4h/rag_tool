#!/bin/bash
# Global environment variables
set -e

# GCP project & location
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ZONE="us-central1-a"

# Service names
export API_SERVICE_NAME="backend-api"

# Container image
export API_IMAGE="gcr.io/${PROJECT_ID}/${API_SERVICE_NAME}:latest"

# Service accounts
export CICD_SA_NAME="cicd-sa"
export CICD_SA="${CICD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export API_SA_NAME="backend-api-sa"
export API_SA="${API_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export VECTORSTORE_SA_NAME="qdrant-vm-sa"
export VECTORSTORE_SA="${VECTORSTORE_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# VectorStore VM configuration
export VECTORSTORE_VM_NAME="qdrant-vm"
export VECTORSTORE_VM_MACHINE_TYPE="e2-standard-2"
export VECTORSTORE_VM_DISK_SIZE="20GB"

# VPC and networking
export CONNECTOR_NAME="cr-connector"
export CONNECTOR_RANGE="10.8.0.0/28"
export SUBNET="default"
export FIREWALL_RULE="allow-cloud-run-to-qdrant"

# Secrets
export OPENAI_SECRET_NAME="openai-api-key"
export API_KEY_SECRET_NAME="api-key"

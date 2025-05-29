#!/bin/bash
# Global environment variables

# GCP project & location
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ZONE="us-central1-a"

# Service names
export API_SERVICE_NAME="backend-api"
export VECTORSTORE_SERVICE_NAME="qdrant-vs"

# Container image
export API_IMAGE="gcr.io/${PROJECT_ID}/${API_SERVICE_NAME}:latest"

# Service accounts
export API_SA_NAME="backend-api-sa"
export API_SA="${API_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export VECTORSTORE_SA_NAME="qdrant-vm-sa"
export VECTORSTORE_SA="${VECTORSTORE_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# VECTORSTORE VM
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

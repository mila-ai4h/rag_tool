#!/bin/bash
# Create and manage IAM service accounts and permissions for GCP project
set -e

# Source the environment variables from the root-based path
source "gcp_assets/config/env.sh"

echo "Setting up service accounts and permissions for project: ${PROJECT_ID}"

# Helper function to create a service account if it doesn't exist
create_sa() {
  local sa_email="$1"
  local sa_name="$2"
  local description="$3"
  if ! gcloud iam service-accounts describe "${sa_email}" --project="${PROJECT_ID}" > /dev/null 2>&1; then
      echo "Creating service account: ${sa_email} (${sa_name})"
      gcloud iam service-accounts create "${sa_name}" \
          --description="${description}" \
          --display-name="${sa_name}"
  else
      echo "Service account ${sa_email} already exists."
  fi
}

# Create and attach roles to API service account
create_sa "${API_SA}" "${API_SA_NAME}" "Service account for API service"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${API_SA}" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${API_SA}" --role="roles/logging.logWriter"

# Create and attach roles to VectorStore account
create_sa "${VECTORSTORE_SA}" "${VECTORSTORE_SA_NAME}" "Service account for Qdrant VM"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${VECTORSTORE_SA}" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${VECTORSTORE_SA}" --role="roles/logging.logWriter"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${VECTORSTORE_SA}" --role="roles/monitoring.metricWriter"

# Create and attach roles to CI/CD service account
create_sa "${CICD_SA}" "${CICD_SA_NAME}" "Service account for GitHub Actions CI/CD pipeline"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${CICD_SA}" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${CICD_SA}" --role="roles/viewer"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${CICD_SA}" --role="roles/run.admin"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${CICD_SA}" --role="roles/iam.serviceAccountUser"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${CICD_SA}" --role="roles/storage.objectCreator"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${CICD_SA}" --role="roles/cloudbuild.builds.editor"



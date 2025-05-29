#!/bin/bash
set -e

# Source the environment variables from the root-based path
source "gcp_assets/config/env.sh"

echo "Cloud Run URL :"
gcloud run services describe "${API_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)'

echo -e "\nPour tester la santé :"
echo "curl -H \"x-api-key: \$API_KEY\" \$(gcloud run services describe ${API_SERVICE_NAME} --region ${REGION} --format='value(status.url)')/health"

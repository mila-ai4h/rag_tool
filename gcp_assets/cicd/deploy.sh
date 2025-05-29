#!/bin/bash
# Deploy GCP resources
set -e

# Source the environment variables
source "gcp_assets/config/env.sh"


echo "Deploying services to project: ${PROJECT_ID}"

# ───────────────────────────────────────────────
# 1. Qdrant VM
# NOTE: assume that we use a pre-built Qdrant image from Docker Hub: qdrant/qdrant
# ───────────────────────────────────────────────
if ! gcloud compute instances describe "${VECTORSTORE_VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" &> /dev/null; then
  echo "Creating Qdrant VM..."
  gcloud compute instances create "${VECTORSTORE_VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${VECTORSTORE_VM_MACHINE_TYPE}" \
    --subnet="default" \
    --no-address \
    --tags=qdrant \
    --boot-disk-type=pd-ssd \
    --boot-disk-size="${VECTORSTORE_VM_DISK_SIZE}" \
    --metadata=startup-script='#!/bin/bash
      set -e
      sudo apt-get update
      sudo apt-get install -y docker.io
      sudo systemctl enable --now docker
      sudo docker run -d --name qdrant --restart=unless-stopped \
        -p 6333:6333 -p 6334:6334 \
        -v /var/lib/qdrant:/qdrant/storage \
        qdrant/qdrant
    '
else
  echo "Qdrant VM already exists, skipping."
fi


# ───────────────────────────────────────────────
# 2. Build Backend API image
# ───────────────────────────────────────────────
echo "Building container image with Cloud Build..."
gcloud builds submit \
  --config "cloudbuild.yaml" \
  --substitutions=_PROJECT_ID="${PROJECT_ID}",_API_SERVICE_NAME="${API_SERVICE_NAME}"
echo "Build completed."

# ───────────────────────────────────────────────
# 3. Deploy API to Cloud Run
# ───────────────────────────────────────────────
echo "Deploying API to Cloud Run..."
PRIVATE_IP=$(gcloud compute instances describe "${VECTORSTORE_VM_NAME}" --zone="${ZONE}" --format='value(networkInterfaces[0].networkIP)')
OPENAI_API_KEY=$(gcloud secrets versions access latest --secret="${OPENAI_SECRET_NAME}")
API_KEY=$(gcloud secrets versions access latest --secret="${API_KEY_SECRET_NAME}")

gcloud run deploy "${API_SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${API_IMAGE}" \
  --vpc-connector="${CONNECTOR_NAME}" \
  --vpc-egress=all-traffic \
  --set-env-vars="QDRANT_HOST=${PRIVATE_IP},QDRANT_PORT=${QDRANT_PORT},OPENAI_API_KEY=${OPENAI_API_KEY},API_KEY=${API_KEY}" \
  --service-account="${API_SA}" \
  --port=8080

echo -e "Deployment completed successfully."

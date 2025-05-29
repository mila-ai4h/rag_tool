#!/bin/bash
set -e

# Source the environment variables from the root-based path
source "gcp_assets/config/env.sh"

echo "Deploying services to project: ${PROJECT_ID} in region: ${REGION}"

gcloud compute routers create nat-router \
  --network=default \
  --region=us-central1

gcloud compute routers nats create nat-config \
  --router=nat-router \
  --region=us-central1 \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips

# Deploy Qdrant on VM
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

# Create VPC connector for Cloud Run
gcloud compute networks vpc-access connectors create "${CONNECTOR_NAME}" \
  --project="${PROJECT_ID}" \
  --network="${SUBNET}" \
  --region="${REGION}" \
  --range="${CONNECTOR_RANGE}"


# Create firewall rule to allow Cloud Run to access Qdrant VM
gcloud compute firewall-rules create "${FIREWALL_RULE}" \
  --project="${PROJECT_ID}" \
  --network="${SUBNET}" \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:6333,tcp:6334 \
  --source-ranges="${CONNECTOR_RANGE}" \
  --target-tags=qdrant


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

PRIVATE_IP=$(gcloud compute instances describe "${VECTORSTORE_VM_NAME}" \
  --zone="${ZONE}" \
  --format='value(networkInterfaces[0].networkIP)')
echo "IP privée de Qdrant : ${PRIVATE_IP}"

echo "Deploying API service to Cloud Run..."
gcloud run deploy "${API_SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${API_IMAGE}" \
  --vpc-connector="${CONNECTOR_NAME}" \
  --vpc-egress=all-traffic \
  --set-env-vars="QDRANT_HOST=${PRIVATE_IP},QDRANT_PORT=${QDRANT_PORT},OPENAI_API_KEY=${OPENAI_API_KEY},API_KEY=dev_key_123" \
  --service-account="${API_SA}" \
  --port=8080
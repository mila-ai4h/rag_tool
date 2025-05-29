#!/bin/bash
set -e

source "gcp_assets/config/env.sh"

echo "Deploying services to project: ${PROJECT_ID} in region: ${REGION}"

# ───────────────────────────────────────────────
# 1. NAT Router
# ───────────────────────────────────────────────
if ! gcloud compute routers describe nat-router --region="${REGION}" --project="${PROJECT_ID}" &> /dev/null; then
  echo "Creating NAT router..."
  gcloud compute routers create nat-router \
    --network=default \
    --region="${REGION}"
else
  echo "NAT router already exists, skipping."
fi

if ! gcloud compute routers nats describe nat-config --router=nat-router --region="${REGION}" --project="${PROJECT_ID}" &> /dev/null; then
  echo "Creating NAT config..."
  gcloud compute routers nats create nat-config \
    --router=nat-router \
    --region="${REGION}" \
    --nat-all-subnet-ip-ranges \
    --auto-allocate-nat-external-ips
else
  echo "NAT config already exists, skipping."
fi

# ───────────────────────────────────────────────
# 2. Qdrant VM
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
# 3. VPC Serverless Connector
# ───────────────────────────────────────────────
if ! gcloud compute networks vpc-access connectors describe "${CONNECTOR_NAME}" --region="${REGION}" --project="${PROJECT_ID}" &> /dev/null; then
  echo "Creating VPC connector..."
  gcloud compute networks vpc-access connectors create "${CONNECTOR_NAME}" \
    --project="${PROJECT_ID}" \
    --network="${SUBNET}" \
    --region="${REGION}" \
    --range="${CONNECTOR_RANGE}"
else
  echo "VPC connector already exists, skipping."
fi

# ───────────────────────────────────────────────
# 4. Firewall Rule
# ───────────────────────────────────────────────
if ! gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project="${PROJECT_ID}" &> /dev/null; then
  echo "Creating firewall rule..."
  gcloud compute firewall-rules create "${FIREWALL_RULE}" \
    --project="${PROJECT_ID}" \
    --network="${SUBNET}" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:6333,tcp:6334 \
    --source-ranges="${CONNECTOR_RANGE}" \
    --target-tags=qdrant
else
  echo "Firewall rule already exists, skipping."
fi

# ───────────────────────────────────────────────
# 5. Build Backend API image
# ───────────────────────────────────────────────
echo "Building container image with Cloud Build..."
gcloud builds submit \
  --config "cloudbuild.yaml" \
  --substitutions=_PROJECT_ID="${PROJECT_ID}",_API_SERVICE_NAME="${API_SERVICE_NAME}"
echo "Build completed."

# ───────────────────────────────────────────────
# 6. Deploy API to Cloud Run
# ───────────────────────────────────────────────
echo "Deploying API to Cloud Run..."
OPENAI_API_KEY=$(gcloud secrets versions access latest --secret="${OPENAI_SECRET_NAME}")
PRIVATE_IP=$(gcloud compute instances describe "${VECTORSTORE_VM_NAME}" --zone="${ZONE}" --format='value(networkInterfaces[0].networkIP)')

gcloud run deploy "${API_SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${API_IMAGE}" \
  --vpc-connector="${CONNECTOR_NAME}" \
  --vpc-egress=all-traffic \
  --set-env-vars="QDRANT_HOST=${PRIVATE_IP},QDRANT_PORT=${QDRANT_PORT},OPENAI_API_KEY=${OPENAI_API_KEY},API_KEY=dev_key_123" \
  --service-account="${API_SA}" \
  --port=8080

echo -e "Deployment completed successfully."

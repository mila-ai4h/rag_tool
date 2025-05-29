#!/bin/bash
# Setup script for GCP project infrastructure
set -e

# Source the environment variables from the root-based path
source "gcp_assets/config/env.sh"

# Bootstrap infrastructure for the project
echo "Bootstrapping infrastructure for project: ${PROJECT_ID} in region: ${REGION}"

# ───────────────────────────────────────────────
# 1. Enable Required APIs
# ───────────────────────────────────────────────
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com \
                        cloudbuild.googleapis.com \
                        secretmanager.googleapis.com \
                        compute.googleapis.com \
                        vpcaccess.googleapis.com

# ───────────────────────────────────────────────
# 2. Set secrets
# ───────────────────────────────────────────────
echo "Creating secrets:"
if ! gcloud secrets describe ${OPENAI_SECRET_NAME} > /dev/null 2>&1; then
    echo "Creating secret: ${OPENAI_SECRET_NAME}"
    gcloud secrets create ${OPENAI_SECRET_NAME}
    echo "Please enter the OpenAI API key:"
    read -s OPENAI_API_KEY
    echo -n ${OPENAI_API_KEY} | gcloud secrets versions add ${OPENAI_SECRET_NAME} --data-file=-
else
    echo "Secret ${OPENAI_SECRET_NAME} already exists."
fi

if ! gcloud secrets describe ${API_KEY_SECRET_NAME} > /dev/null 2>&1; then
    echo "Creating secret: ${API_KEY_SECRET_NAME}"
    gcloud secrets create ${API_KEY_SECRET_NAME}
    echo "Please enter the API key:"
    read -s API_KEY
    echo -n ${API_KEY} | gcloud secrets versions add ${API_KEY_SECRET_NAME} --data-file=-
else
    echo "Secret ${API_KEY_SECRET_NAME} already exists."
fi

# ───────────────────────────────────────────────
# 3. NAT Router
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
# 4. VPC Serverless Connector
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
# 5. Firewall Rule
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
#!/usr/bin/env bash
set -euo pipefail

#######################################################################
# 1. Variables à personnaliser
#######################################################################
PROJECT_ID="my-project"
REGION="us-central1"            # même région que Cloud Run
ZONE="us-central1-a"            # zone de la VM
SUBNET="default"                # ou un sous-réseau dédié
VM_NAME="qdrant-vm"
VM_MACHINE="e2-standard-4"
VM_DISK_SIZE="20GB"             # SSD conseillé
CONNECTOR_NAME="cr-connector"
CONNECTOR_RANGE="10.8.0.0/28"   # plage CIDR libre
FIREWALL_RULE="allow-qdrant"
API_SERVICE_NAME="backend-api"
API_IMAGE="gcr.io/${PROJECT_ID}/backend:latest"    # ou Artifact Registry
API_SA="cloud-run-sa@${PROJECT_ID}.iam.gserviceaccount.com"
OPENAI_SECRET_NAME="openai-api-key"
QDRANT_PORT=6333

#######################################################################
# 2. Pré-requis : activer les API nécessaires
#######################################################################
gcloud services enable \
  compute.googleapis.com \
  run.googleapis.com \
  vpcaccess.googleapis.com \
  cloudbuild.googleapis.com

#######################################################################
# 3. (Optionnel) construire l’image du backend
#######################################################################
# gcloud builds submit --tag "${API_IMAGE}" .

#######################################################################
# 4. Créer la VM Qdrant (sans IP publique)
#######################################################################
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${VM_MACHINE}" \
  --subnet="${SUBNET}" \
  --no-address \
  --tags=qdrant \
  --boot-disk-type=pd-ssd \
  --boot-disk-size="${VM_DISK_SIZE}" \
  --metadata=startup-script='#!/bin/bash
    set -e
    apt-get update
    apt-get install -y docker.io
    systemctl enable --now docker
    docker run -d --name qdrant --restart=unless-stopped \
      -p 6333:6333 -p 6334:6334 \
      -v /var/lib/qdrant:/qdrant/storage \
      qdrant/qdrant
  '

#######################################################################
# 5. Créer le connector Serverless VPC Access
#######################################################################
gcloud compute networks vpc-access connectors create "${CONNECTOR_NAME}" \
  --project="${PROJECT_ID}" \
  --network="${SUBNET}" \
  --region="${REGION}" \
  --range="${CONNECTOR_RANGE}"

#######################################################################
# 6. Règle pare-feu : Cloud Run ➜ Qdrant
#######################################################################
gcloud compute firewall-rules create "${FIREWALL_RULE}" \
  --project="${PROJECT_ID}" \
  --network="${SUBNET}" \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:6333,tcp:6334 \
  --source-ranges="${CONNECTOR_RANGE}" \
  --target-tags=qdrant

#######################################################################
# 7. Récupérer l’IP interne de la VM
#######################################################################
PRIVATE_IP=$(gcloud compute instances describe "${VM_NAME}" \
  --zone="${ZONE}" \
  --format='value(networkInterfaces[0].networkIP)')

echo "IP privée de Qdrant : ${PRIVATE_IP}"

#######################################################################
# 8. Récupérer la clé OpenAI stockée dans Secret Manager
#######################################################################
OPENAI_API_KEY=$(gcloud secrets versions access latest \
  --project="${PROJECT_ID}" \
  --secret="${OPENAI_SECRET_NAME}")

#######################################################################
# 9. Déployer le backend sur Cloud Run
#######################################################################
gcloud run deploy "${API_SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${API_IMAGE}" \
  --vpc-connector="${CONNECTOR_NAME}" \
  --egress-settings=all \
  --set-env-vars="QDRANT_HOST=${PRIVATE_IP},QDRANT_PORT=${QDRANT_PORT},OPENAI_API_KEY=${OPENAI_API_KEY}" \
  --service-account="${API_SA}" \
  --port=8080

#######################################################################
# 10. Vérification rapide
#######################################################################
echo -e "\n✅ Déploiement terminé.\n"
echo "Cloud Run URL :"
gcloud run services describe "${API_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)'

echo -e "\nPour tester la santé :"
echo "curl -H \"x-api-key: \$API_KEY\" \$(gcloud run services describe ${API_SERVICE_NAME} --region ${REGION} --format='value(status.url)')/health"

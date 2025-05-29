#!/bin/bash
set -e

# Source the environment variables from the root-based path
source "gcp_assets/config/env.sh"

# Bootstrap infrastructure for the project
echo "Bootstrapping infrastructure for project: ${PROJECT_ID} in region: ${REGION}"

# Enable required APIs for resource creation
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com \
                        cloudbuild.googleapis.com \
                        secretmanager.googleapis.com \
                        compute.googleapis.com \
                        vpcaccess.googleapis.com

# Creation secrets ask user of input (openai-api-key)
echo "Creating secrets: ${OPENAI_SECRET_NAME}"
if ! gcloud secrets describe ${OPENAI_SECRET_NAME} > /dev/null 2>&1; then
    echo "Creating secret: ${OPENAI_SECRET_NAME}"
    gcloud secrets create ${OPENAI_SECRET_NAME}
    echo "Please enter the OpenAI API key:"
    read -s OPENAI_API_KEY
    echo -n ${OPENAI_API_KEY} | gcloud secrets versions add ${OPENAI_SECRET_NAME} --data-file=-
else
    echo "Secret ${OPENAI_SECRET_NAME} already exists."
fi
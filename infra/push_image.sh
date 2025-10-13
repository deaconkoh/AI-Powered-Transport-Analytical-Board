#!/bin/bash
set -eux

# Variables
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1"
REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/traffic-ai-repo"

# Build the Docker image
docker build -t traffic-ai .

# Tag the image with 'latest' or custom tag argument
TAG=${1:-latest}
docker tag traffic-ai:latest $REPO:$TAG

# Authenticate and push to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REPO
docker push $REPO:$TAG

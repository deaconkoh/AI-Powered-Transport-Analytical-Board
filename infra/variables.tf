variable "project_name" {
  type    = string
  default = "traffic-ai"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "ec2_role_name" {
  type        = string
  description = "IAM role name attached to EC2 instance that can invoke SageMaker endpoint"
  default     = ""
}

variable "google_maps_api_key" {
  type        = string
  sensitive   = true
  description = "Google Maps API key (optional, used for local testing or secret storage)"
  default     = ""
}

# Model artifact (created at repo root as model.tar.gz)
variable "model_archive_path" {
  type        = string
  description = "Local path to the model tarball (relative to infra/)."
  default     = "../model.tar.gz"
}

variable "model_key" {
  type        = string
  description = "S3 key to upload the model tarball to."
  default     = "ensemble/model.tar.gz"
}

# SageMaker serverless config (fits your current quota)
variable "serverless_memory_mb" {
  type    = number
  default = 3072
}
variable "serverless_max_conc" {
  type    = number
  default = 4
}

# Inference container (AWS DLC) – us-east-1 CPU
variable "image_uri" {
  type    = string
  default = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
}

variable "raw_bucket_name" {
  type        = string
  description = "S3 bucket that holds raw ingestion (e.g., raw/lta/CarParkAvailability)"
}

variable "enable_ingestion" {
  type        = bool
  default     = true
  description = "Attach EventBridge targets to Lambda (on/off)."
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
} # ALB here

variable "private_subnet_ids" {
  type = list(string)
}

variable "alb_cert_arn" {
  type        = string
  default     = ""
  description = "ACM cert ARN for HTTPS. Leave blank to disable HTTPS."
}


variable "enable_single_ec2" {
  type    = bool
  default = false
}
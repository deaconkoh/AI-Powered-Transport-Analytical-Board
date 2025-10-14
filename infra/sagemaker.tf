############################
# Helpers
############################
data "aws_caller_identity" "current" {}

locals {
  bucket_name   = "${var.project_name}-models-${data.aws_caller_identity.current.account_id}-${var.region}"
  endpoint_name = "${var.project_name}-serverless"
  model_name    = "${var.project_name}-ensemble-model"
  ep_cfg_name   = "${var.project_name}-serverless-cfg"
}

############################
# S3 bucket for model artifact
############################
resource "aws_s3_bucket" "models" {
  bucket        = local.bucket_name
  force_destroy = true
}
resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id
  rule { object_ownership = "BucketOwnerPreferred" }
}
resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Upload model.tar.gz that you build at repo root
resource "aws_s3_object" "model_artifact" {
  bucket = aws_s3_bucket.models.id
  key    = var.model_key
  source = var.model_archive_path
  etag   = filemd5(var.model_archive_path)
}

############################
# SageMaker execution role
############################
resource "aws_iam_role" "sm_exec" {
  name = "${var.project_name}-sm-exec-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{ Effect = "Allow", Principal = { Service = "sagemaker.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

# Least-privilege S3 read for the model
resource "aws_iam_policy" "sm_s3_read" {
  name = "${var.project_name}-sm-s3-read"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Action    = ["s3:ListBucket"],
        Resource  = "arn:aws:s3:::${local.bucket_name}",
        Condition = { StringLike = { "s3:prefix" : ["${var.model_key}", "${dirname(var.model_key)}/*"] } }
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject"],
        Resource = "arn:aws:s3:::${local.bucket_name}/${var.model_key}"
      }
    ]
  })
}
resource "aws_iam_role_policy_attachment" "sm_s3_read_attach" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = aws_iam_policy.sm_s3_read.arn
}

resource "aws_iam_policy" "sm_secrets_read" {
  name = "${var.project_name}-sm-secrets-read"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["secretsmanager:GetSecretValue"],
        Resource = "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:traffic-ai-app-secrets-*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sm_secrets_read_attach" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = aws_iam_policy.sm_secrets_read.arn
}


resource "aws_iam_role_policy_attachment" "sm_full" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

############################
# SageMaker model & endpoint (serverless)
############################
resource "aws_sagemaker_model" "ensemble" {
  name               = local.model_name
  execution_role_arn = aws_iam_role.sm_exec.arn

  primary_container {
    image          = var.image_uri
    mode           = "SingleModel"
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/${aws_s3_object.model_artifact.key}"
    environment = {
      SAGEMAKER_PROGRAM          = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "/opt/ml/model/code"
      USE_LIVE_APIS              = "false" # model won’t call LTA/NEA
      AWS_DEFAULT_REGION         = "us-east-1"
      AWS_REGION                 = "us-east-1"
      APP_SECRET_NAME            = "traffic-ai-app-secrets"
    }
  }

  depends_on = [aws_s3_object.model_artifact]
}

resource "aws_sagemaker_endpoint_configuration" "serverless_cfg" {
  name = local.ep_cfg_name

  production_variants {
    model_name   = aws_sagemaker_model.ensemble.name
    variant_name = "AllTraffic"

    serverless_config {
      memory_size_in_mb = var.serverless_memory_mb
      max_concurrency   = var.serverless_max_conc
    }
  }
}

resource "aws_sagemaker_endpoint" "serverless" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.serverless_cfg.name
}

output "sm_endpoint_name" { value = aws_sagemaker_endpoint.serverless.name }

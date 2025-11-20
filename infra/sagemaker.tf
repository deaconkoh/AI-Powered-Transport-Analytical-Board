################################################################################
# 1. HELPERS & LOCALS
################################################################################
data "aws_caller_identity" "current" {}

locals {
  # Naming conventions
  bucket_name              = "${var.project_name}-models-${data.aws_caller_identity.current.account_id}-${var.region}"
  endpoint_name            = "${var.project_name}-serverless"
  model_name               = "${var.project_name}-ensemble-model"
  ep_cfg_name              = "${var.project_name}-serverless-cfg"
  
  # File paths
  hybrid_training_code_key = "code/hybrid-training.tar.gz"
}

################################################################################
# 2. S3 BUCKET (MODELS & ARTIFACTS)
################################################################################
resource "aws_s3_bucket" "models" {
  bucket        = local.bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle: Clean up temp training output after 30 days (Cost savings)
resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    id     = "cleanup-temp-training-output"
    status = "Enabled"
    filter {
      prefix = "hybrid/training-output/" 
    }
    expiration {
      days = 30 
    }
  }
}

################################################################################
# 3. S3 OBJECTS (INITIAL UPLOADS)
################################################################################
# Upload existing ensemble model artifact
resource "aws_s3_object" "model_artifact" {
  bucket = aws_s3_bucket.models.id
  key    = var.model_key
  source = var.model_archive_path
  etag   = filemd5(var.model_archive_path)
}

# Upload hybrid training code
resource "aws_s3_object" "hybrid_training_code" {
  bucket = aws_s3_bucket.models.id
  key    = local.hybrid_training_code_key
  source = var.hybrid_training_code_path
  etag   = filemd5(var.hybrid_training_code_path)
}

################################################################################
# 4. SSM PARAMETERS (REGISTRY & METRICS)
################################################################################
resource "aws_ssm_parameter" "hybrid_current_model_uri" {
  name      = "/traffic/hybrid/current_model_uri"
  type      = "String"
  value     = "s3://${aws_s3_bucket.models.id}/hybrid/training-output/traffic-ai-hybrid-20251118-202441//output/model.tar.gz"
  
  overwrite = true

  # -------------------------------------------------------
  # Ignore changes so Terraform doesn't undo your 
  # future automated updates from the Registry Lambda
  # -------------------------------------------------------
  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "hybrid_current_metric" {
  name      = "/traffic/hybrid/current_metric"
  type      = "String"
  value     = "9999.9" # High initial error so first training always wins
  overwrite = true
}

################################################################################
# 5. IAM ROLE (SAGEMAKER EXECUTION)
################################################################################
resource "aws_iam_role" "sm_exec" {
  name = "${var.project_name}-sm-exec-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{ 
      Effect = "Allow", 
      Principal = { Service = "sagemaker.amazonaws.com" }, 
      Action = "sts:AssumeRole" 
    }]
  })
}

# --- Policy 1: Read/Write to Models Bucket ---
resource "aws_iam_policy" "sm_models_rw" {
  name = "${var.project_name}-sm-models-rw"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["s3:ListBucket"],
        Resource = aws_s3_bucket.models.arn
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject", "s3:PutObject"],
        Resource = "${aws_s3_bucket.models.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sm_models_attach" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = aws_iam_policy.sm_models_rw.arn
}

# --- Policy 2: Read from RAW Bucket (Gold Data) ---
resource "aws_iam_policy" "sm_raw_read" {
  name = "${var.project_name}-sm-raw-read"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["s3:ListBucket"],
        Resource = "arn:aws:s3:::${var.raw_bucket}",
        Condition = {
          StringLike = { "s3:prefix" : ["gold/speedbands/*"] }
        }
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject"],
        Resource = "arn:aws:s3:::${var.raw_bucket}/gold/speedbands/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sm_raw_read_attach" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = aws_iam_policy.sm_raw_read.arn
}

# --- Policy 3: Read Secrets & SSM ---
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

resource "aws_iam_policy" "sm_ssm_read" {
  name = "${var.project_name}-sm-ssm-read"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["ssm:GetParameter", "ssm:GetParameters"],
        Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/traffic/hybrid/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sm_ssm_read_attach" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = aws_iam_policy.sm_ssm_read.arn
}

# --- Policy 4: Full Access (Logs, Metrics, etc) ---
resource "aws_iam_role_policy_attachment" "sm_full" {
  role       = aws_iam_role.sm_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

################################################################################
# 6. SAGEMAKER MODEL & ENDPOINT (SERVERLESS)
################################################################################
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
      USE_LIVE_APIS              = "false"
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

################################################################################
# 7. OUTPUTS
################################################################################
output "sm_endpoint_name" {
  value = aws_sagemaker_endpoint.serverless.name
}
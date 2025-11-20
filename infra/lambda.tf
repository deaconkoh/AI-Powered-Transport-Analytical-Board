############################
# S3 bucket to land raw data
############################
data "aws_caller_identity" "me" {}
locals {
  raw_bucket_name = "${var.project_name}-raw-${data.aws_caller_identity.me.account_id}-${var.region}"
  app_secret_name = "${var.project_name}-app-secrets"
}

resource "aws_s3_bucket" "raw" {
  bucket        = local.raw_bucket_name
  force_destroy = true
}

# Enable versioning on the RAW bucket for point-in-time recovery
resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle rule to clean up very old object versions
resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  #############################################
  # 1. Retain only 1 day of CARPARK raw data
  #############################################
  rule {
    id     = "expire-carpark-raw"
    status = "Enabled"

    filter {
      prefix = "raw/lta/carparks/"
    }

    expiration {
      days = 1
    }
  }

  #############################################
  # 2. Retain BRONZE speedbands for 365 days
  #    Keep older versions for 30 days (you already set this)
  #############################################
  rule {
    id     = "expire-speedbands-bronze"
    status = "Enabled"

    filter {
      prefix = "raw/lta/speedbands/"
    }

    expiration {
      days = 365    # 1 year of raw history
    }
  }

  #############################################
  # 3. Silver (curated) — keep 1 year only
  #############################################
  rule {
    id     = "expire-speedbands-silver"
    status = "Enabled"

    filter {
      prefix = "silver/speedbands/"
    }

    expiration {
      days = 365
    }
  }

  #############################################
  # 4. Gold (ML features) — keep 1 year only
  #############################################
  rule {
    id     = "expire-speedbands-gold"
    status = "Enabled"

    filter {
      prefix = "gold/speedbands/"
    }

    expiration {
      days = 365
    }
  }

  #############################################
  # 5. NONCURRENT VERSION CLEANUP
  #    Applies bucket-wide (affects all prefixes)
  #    Already in your config — we keep it
  #############################################
  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30   # delete old versions after 30 days
    }
  }
}


resource "aws_s3_bucket_public_access_block" "raw" {
  bucket                  = aws_s3_bucket.raw.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

############################
# Lambda execution role
############################
resource "aws_iam_role" "ingest_role" {
  name = "${var.project_name}-lambda-ingest-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "lambda.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

# Logs + VPC basic permissions
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.ingest_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Allow writing to our RAW bucket
resource "aws_iam_policy" "raw_put" {
  name = "${var.project_name}-lambda-raw-put"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = ["s3:PutObject", "s3:AbortMultipartUpload", "s3:PutObjectAcl"],
      Resource = "arn:aws:s3:::${local.raw_bucket_name}/*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "raw_put_attach" {
  role       = aws_iam_role.ingest_role.name
  policy_arn = aws_iam_policy.raw_put.arn
}

# Allow reading the app secret
data "aws_secretsmanager_secret" "app" {
  name = local.app_secret_name
}

resource "aws_iam_policy" "secrets_read" {
  name = "${var.project_name}-lambda-secrets-read"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = ["secretsmanager:GetSecretValue"],
      Resource = data.aws_secretsmanager_secret.app.arn
    }]
  })
}

resource "aws_iam_role_policy_attachment" "secrets_read_attach" {
  role       = aws_iam_role.ingest_role.name
  policy_arn = aws_iam_policy.secrets_read.arn
}

# --- ADD TO infra/lambda.tf ---

# 1. Zip the Python Code
data "archive_file" "registry_zip" {
  type        = "zip"
  source_file = "${path.module}/../src/backend/ops/register_model.py" 
  output_path = "${path.module}/build/registry_lambda.zip"
}

# 2. The Registry Lambda Function
resource "aws_lambda_function" "registry" {
  filename         = data.archive_file.registry_zip.output_path
  function_name    = "${var.project_name}-model-registry"
  role             = aws_iam_role.registry_role.arn
  handler          = "register_model.lambda_handler"
  source_code_hash = data.archive_file.registry_zip.output_base64sha256
  runtime          = "python3.10"
  timeout          = 60

  environment {
    variables = {
      MODELS_BUCKET = aws_s3_bucket.models.id
    }
  }
  
  # AWS Data Wrangler Layer (for Pandas/Numpy support)
  layers = ["arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python310:12"]
}

# 3. IAM Role for this Lambda
resource "aws_iam_role" "registry_role" {
  name = "${var.project_name}-registry-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" } }]
  })
}

resource "aws_iam_policy" "registry_policy" {
  name = "${var.project_name}-registry-policy"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
            # S3 & Logs
            "s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:CopyObject",
            "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
            
            # SSM
            "ssm:GetParameter", "ssm:PutParameter",
            
            # SageMaker (Monitoring & Registry need these)
            "sagemaker:DescribeTrainingJob",
            "sagemaker:CreateTrainingJob",
            
            # IAM (Required to pass the execution role to SageMaker)
            "iam:PassRole"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "registry_attach" {
  role       = aws_iam_role.registry_role.name
  policy_arn = aws_iam_policy.registry_policy.arn
}

# 4. EventBridge Trigger (SageMaker -> Lambda)
resource "aws_cloudwatch_event_rule" "sm_success" {
  name        = "${var.project_name}-capture-training-success"
  description = "Trigger Registry when SageMaker Training completes"

  event_pattern = jsonencode({
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Training Job State Change"],
    "detail": {
      "TrainingJobStatus": ["Completed"]
    }
  })
}

resource "aws_cloudwatch_event_target" "trigger_registry" {
  rule      = aws_cloudwatch_event_rule.sm_success.name
  target_id = "SendToLambda"
  arn       = aws_lambda_function.registry.arn
}

resource "aws_lambda_permission" "allow_eventbridge_registry" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.registry.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sm_success.arn
}

############################
# Package Lambda code from local files
############################
data "archive_file" "lta_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda-ingest/lta_ingest.py"
  output_path = "${path.module}/build/lta_ingest.zip"
}

data "archive_file" "nea_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda-ingest/nea_ingest.py"
  output_path = "${path.module}/build/nea_ingest.zip"
}

############################
# Lambda functions
############################
resource "aws_lambda_function" "lta_ingest" {
  function_name    = "${var.project_name}-lta-ingest"
  role             = aws_iam_role.ingest_role.arn
  runtime          = "python3.11"
  handler          = "lta_ingest.lambda_handler"
  filename         = data.archive_file.lta_zip.output_path
  source_code_hash = filebase64sha256(data.archive_file.lta_zip.output_path)
  timeout          = 900
  memory_size  = 1024

  environment {
    variables = {
      RAW_BUCKET      = aws_s3_bucket.raw.bucket
      APP_SECRET_NAME = local.app_secret_name
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic]
}

resource "aws_lambda_function" "nea_ingest" {
  function_name    = "${var.project_name}-nea-ingest"
  role             = aws_iam_role.ingest_role.arn
  runtime          = "python3.11"
  handler          = "nea_ingest.lambda_handler"
  filename         = data.archive_file.nea_zip.output_path
  source_code_hash = filebase64sha256(data.archive_file.nea_zip.output_path)
  timeout          = 60

  environment {
    variables = {
      RAW_BUCKET      = aws_s3_bucket.raw.bucket
      APP_SECRET_NAME = local.app_secret_name
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic]
}

############################
# Schedules (EventBridge)
############################
# LTA every 5 minutes
resource "aws_cloudwatch_event_rule" "lta_rule" {
  name                = "${var.project_name}-lta-every-5m"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "lta_target" {
  count     = var.enable_ingestion ? 1 : 0
  rule      = aws_cloudwatch_event_rule.lta_rule.name
  target_id = "lta-lambda"
  arn       = aws_lambda_function.lta_ingest.arn
}

resource "aws_lambda_permission" "lta_invoke" {
  # count         = var.enable_ingestion ? 1 : 0
  count     = 1
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lta_ingest.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.lta_rule.arn
}

# NEA every 10 minutes
resource "aws_cloudwatch_event_rule" "nea_rule" {
  name                = "${var.project_name}-nea-every-10m"
  schedule_expression = "rate(10 minutes)"
}

resource "aws_cloudwatch_event_target" "nea_target" {
  count     = var.enable_ingestion ? 1 : 0
  rule      = aws_cloudwatch_event_rule.nea_rule.name
  target_id = "nea-lambda"
  arn       = aws_lambda_function.nea_ingest.arn
}

resource "aws_lambda_permission" "nea_invoke" {
  count         = var.enable_ingestion ? 1 : 0
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.nea_ingest.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.nea_rule.arn
}

resource "aws_glue_trigger" "speedband_etl_schedule" {
  count    = var.enable_glue ? 1 : 0
  name     = "${var.project_name}-speedband-etl-every-15m"
  type     = "SCHEDULED"

  schedule = "cron(0 0 * * ? *)"

  actions {
    job_name = aws_glue_job.speedband_etl.name
  }

  enabled = true
}

# --- Monitor Lambda (Auto-Retrain) ---

# Zip Code
data "archive_file" "monitor_zip" {
  type        = "zip"
  source_file = "${path.module}/../src/backend/ops/monitor_model.py"
  output_path = "${path.module}/build/monitor_lambda.zip"
}

# Function
resource "aws_lambda_function" "monitor" {
  function_name    = "${var.project_name}-model-monitor"
  role             = aws_iam_role.registry_role.arn 
  handler          = "monitor_model.lambda_handler"
  runtime          = "python3.10"
  filename         = data.archive_file.monitor_zip.output_path
  source_code_hash = data.archive_file.monitor_zip.output_base64sha256
  timeout          = 300  # 5 minutes
  memory_size      = 1024 # 1 GB RAM

  environment {
    variables = {
      PROJECT_NAME  = var.project_name
      ROLE_ARN      = aws_iam_role.sm_exec.arn
      MODELS_BUCKET = aws_s3_bucket.models.id
      RAW_BUCKET    = aws_s3_bucket.raw.id
    }
  }

  layers = ["arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python310:12"]
}

# Schedule (Run Daily at 8 AM)
resource "aws_cloudwatch_event_rule" "monitor_schedule" {
  name                = "${var.project_name}-monitor-daily"
  schedule_expression = "cron(0 8 * * ? *)"
}

resource "aws_cloudwatch_event_target" "monitor_target" {
  rule      = aws_cloudwatch_event_rule.monitor_schedule.name
  target_id = "monitor-lambda"
  arn       = aws_lambda_function.monitor.arn
}

resource "aws_lambda_permission" "allow_eventbridge_monitor" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.monitor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.monitor_schedule.arn
}

############################
# Outputs
############################
output "raw_bucket" {
  value = aws_s3_bucket.raw.bucket
}
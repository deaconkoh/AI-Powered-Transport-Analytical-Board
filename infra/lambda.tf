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
  timeout          = 60

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
  count         = var.enable_ingestion ? 1 : 0
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

############################
# Outputs
############################
output "raw_bucket" {
  value = aws_s3_bucket.raw.bucket
}

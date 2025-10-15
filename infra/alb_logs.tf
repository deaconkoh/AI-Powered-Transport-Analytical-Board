# Create the S3 bucket for ALB access logs
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "${var.project_name}-alb-logs"
  force_destroy = true

  tags = {
    Name = "${var.project_name}-alb-logs"
  }
}

# Lifecycle configuration (replaces lifecycle_rule block)
resource "aws_s3_bucket_lifecycle_configuration" "alb_logs_lifecycle" {
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

    expiration {
      days = 30
    }
  }
}

# Bucket policy to allow ALB to write logs
resource "aws_s3_bucket_policy" "alb_logs_policy" {
  bucket = aws_s3_bucket.alb_logs.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "AllowALBLogging",
        Effect = "Allow",
        Principal = {
          Service = "logdelivery.elasticloadbalancing.amazonaws.com"
        },
        Action = [
          "s3:PutObject"
        ],
        Resource = "${aws_s3_bucket.alb_logs.arn}/*"
      },
      {
        Sid    = "GetBucketAclPermission",
        Effect = "Allow",
        Principal = {
          Service = "logdelivery.elasticloadbalancing.amazonaws.com"
        },
        Action = [
          "s3:GetBucketAcl"
        ],
        Resource = aws_s3_bucket.alb_logs.arn
      }
    ]
  })
}

resource "aws_athena_database" "traffic_db" {
  name   = "traffic_ai_db"
  bucket = aws_s3_bucket.raw.id
}

# Bucket to store query results (Athena requires this)
resource "aws_s3_bucket" "athena_results" {
  bucket        = "${var.project_name}-athena-results-${data.aws_caller_identity.me.account_id}"
  force_destroy = true
}

# Crawler to find your Batch Predictions
resource "aws_glue_crawler" "predictions_crawler" {
  database_name = aws_athena_database.traffic_db.name
  name          = "${var.project_name}-predictions-crawler"
  role          = aws_iam_role.glue_role.arn

  s3_target {
    # Pointing to where your batch job saves files
    path = "s3://${aws_s3_bucket.raw.id}/predictions/hybrid/"
  }
  
  # Run once every hour to pick up new files
  schedule = "cron(30 * * * ? *)"
}
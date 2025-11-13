resource "aws_s3_object" "speedband_etl_script" {
  bucket = aws_s3_bucket.raw.id
  key    = "scripts/speedband_etl.py"

  # Path from infra/ to your script in src/
  source = "${path.module}/../src/backend/bigdata/speedband_etl.py"

  # Force re-upload when file changes
  etag = filemd5("${path.module}/../src/backend/bigdata/speedband_etl.py")
}


resource "aws_glue_job" "speedband_etl" {
  name     = "${var.project_name}-speedband-etl"
  role_arn = aws_iam_role.glue_role.arn

  command {
    name            = "glueetl"
    script_location = "s3://${local.raw_bucket_name}/scripts/speedband_etl.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"  = "python"
    "--RAW_BUCKET"    = local.raw_bucket_name
    "--enable-metrics" = "true"
  }

  glue_version       = "4.0"
  number_of_workers  = 2
  worker_type        = "G.1X"
}

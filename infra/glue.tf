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

  connections = [aws_glue_connection.private_network.name]

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
  worker_type        = "G.1X"
  number_of_workers  = 2   
}

# ==========================================
# Batch Inference Job (Runs Nightly)
# ==========================================

# 1. Upload the batch inference script (Only if glue is enabled)
resource "aws_s3_object" "batch_script" {
  count  = var.enable_glue ? 1 : 0
  bucket = aws_s3_bucket.models.id
  key    = "code/batch_inference.py"
  source = "${path.module}/../src/backend/ops/batch_inference.py"
  etag   = filemd5("${path.module}/../src/backend/ops/batch_inference.py")
}

# 2. The Glue Job (Python Shell)
resource "aws_glue_job" "batch_inference" {
  count    = var.enable_glue ? 1 : 0
  name     = "${var.project_name}-batch-inference"
  role_arn = aws_iam_role.glue_role.arn 
  
  command {
    name            = "pythonshell"
    python_version  = "3.9"
    script_location = "s3://${aws_s3_bucket.models.id}/code/batch_inference.py"
  }
  
  connections = [aws_glue_connection.private_network.name]

  default_arguments = {
    "--RAW_BUCKET"                = aws_s3_bucket.raw.id
    "--additional-python-modules" = "torch,pandas,numpy,awswrangler"
  }
  
  max_capacity = 1.0
  timeout      = 2880
}

# 3. Schedule: Run at 1 AM UTC (Only if glue is enabled)
resource "aws_glue_trigger" "batch_trigger" {
  count    = var.enable_glue ? 1 : 0
  name     = "${var.project_name}-batch-trigger"
  type     = "SCHEDULED"
  schedule = "cron(0 1 * * ? *)" 
  
  actions {
    job_name = aws_glue_job.batch_inference[0].name
  }
  
  enabled = true
}

resource "aws_glue_connection" "private_network" {
  name = "${var.project_name}-glue-conn"
  connection_type = "NETWORK"

  physical_connection_requirements {
    availability_zone      = "us-east-1a"
    security_group_id_list = [aws_security_group.glue_sg.id]
    subnet_id              = aws_subnet.private_a.id
  }
}
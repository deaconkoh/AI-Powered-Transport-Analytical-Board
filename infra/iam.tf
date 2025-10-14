#############################
# IAM for EC2 -> ECR + Secret (Google only)
#############################

resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ec2.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "raw_s3_read_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.raw_s3_read.arn
}

resource "aws_iam_policy" "raw_s3_read" {
  name = "${var.project_name}-raw-s3-read"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["s3:ListBucket"],
        Resource = "arn:aws:s3:::${aws_s3_bucket.raw.bucket}",
        Condition = {
          StringLike = {
            "s3:prefix" : [
              "raw/lta/carparks/*",
              "raw/lta/carparks/"
            ]
          }
        }
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject"],
        Resource = "arn:aws:s3:::${aws_s3_bucket.raw.bucket}/raw/lta/carparks/*"
      }
    ]
  })
}

# ECR pull
resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Allow EC2 to read ONLY the Google Maps secret
resource "aws_iam_policy" "read_google_secret" {
  name = "${var.project_name}-read-google-secret"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect : "Allow",
      Action : ["secretsmanager:GetSecretValue"],
      Resource : aws_secretsmanager_secret.google_maps.arn
    }]
  })
}
resource "aws_iam_role_policy_attachment" "read_google_secret_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.read_google_secret.arn
}

# Instance profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}
resource "aws_iam_policy" "invoke_sm" {
  name = "${var.project_name}-invoke-sm"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = "sagemaker:InvokeEndpoint",
      Resource = "*" // or the specific endpoint ARN for tighter scope
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_invoke_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.invoke_sm.arn
}


#############################
# Networking / Security Group
#############################
resource "aws_security_group" "traffic_sg" {
  name        = "${var.project_name}-sg"
  description = "Allow HTTP 8080 and SSH access"

  ingress {
    description = "HTTP"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # tighten in prod
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-sg" }
}

#############################
# AMI
#############################
data "aws_ami" "amazon_linux2" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

#############################
# SSH Key Pair (uses local .pub)
#############################
resource "aws_key_pair" "traffic_ai" {
  key_name   = "traffic-ai-key"
  public_key = file(pathexpand("~/.ssh/traffic-ai-key.pub"))
  lifecycle { ignore_changes = [public_key] }
}

#############################
# EC2 Instance
#############################
# NOTE: this references aws_ecr_repository.app and the SageMaker endpoint
# from sagemaker.tf. Keep all in one module so references resolve.
resource "aws_instance" "traffic_ec2" {
  ami                         = data.aws_ami.amazon_linux2.id
  instance_type               = "t3.small"
  key_name                    = aws_key_pair.traffic_ai.key_name
  vpc_security_group_ids      = [aws_security_group.traffic_sg.id]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    iops        = 3000
    throughput  = 125
  }

  lifecycle { create_before_destroy = true }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail

    yum install -y docker awscli jq
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ec2-user

    REGION="${var.region}"
    REPO_URL="${aws_ecr_repository.app.repository_url}"
    IMAGE_TAG="${var.image_tag}"

    # ECR auth + pull
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REPO_URL"
    docker system prune -af || true
    docker pull "$REPO_URL:$IMAGE_TAG"

    # ---- Pull only Google key from Secrets Manager ----
    GOOGLE_SECRET_ID="${aws_secretsmanager_secret.google_maps.name}"
    GOOGLE_SECRET_JSON="$(aws secretsmanager get-secret-value --region "$REGION" --secret-id "$GOOGLE_SECRET_ID" --query SecretString --output text || echo '{}')"
    GOOGLE_MAP_KEY="$(echo "$GOOGLE_SECRET_JSON" | jq -r '.key // empty')"

    # restart container
    docker rm -f traffic-ai || true

    # If you want EC2 to call SageMaker for predictions:
    SM_ENDPOINT="${aws_sagemaker_endpoint.serverless.name}"

    docker run -d --name traffic-ai -p 8080:8080 --restart unless-stopped \
      -e USE_LIVE_APIS=false \
      -e USE_SM=true \
      -e RAW_BUCKET="${aws_s3_bucket.raw.bucket}" \
      -e AWS_REGION="$REGION" \
      -e AWS_DEFAULT_REGION="$REGION" \
      -e SM_ENDPOINT="$SM_ENDPOINT" \
      -e GOOGLE_MAP_KEY="$GOOGLE_MAP_KEY" \
      "$REPO_URL:$IMAGE_TAG"
  EOF

  tags = { Name = "${var.project_name}-ec2" }

  depends_on = [
    aws_ecr_repository.app,
    aws_iam_instance_profile.ec2_profile,
    aws_iam_role_policy_attachment.ecr_readonly,
    aws_iam_role_policy_attachment.read_google_secret_attach,
    aws_sagemaker_endpoint.serverless
  ]
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


resource "aws_iam_role_policy_attachment" "raw_s3_read_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.raw_s3_read.arn
}


#############################
# Elastic IP
#############################
resource "aws_eip" "traffic_ip" { domain = "vpc" }

resource "aws_eip_association" "traffic_assoc" {
  instance_id   = aws_instance.traffic_ec2.id
  allocation_id = aws_eip.traffic_ip.id
}

#############################
# Outputs
#############################
output "ec2_public_ip" { value = aws_eip.traffic_ip.public_ip }
output "ec2_url" { value = "http://${aws_eip.traffic_ip.public_ip}:8080/healthz" }

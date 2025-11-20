#############################
# Networking / Security Group
#############################
resource "aws_security_group" "traffic_sg" {
  count       = var.enable_single_ec2 ? 1 : 0
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
  count      = var.enable_single_ec2 ? 1 : 0
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
  count                       = var.enable_single_ec2 ? 1 : 0
  ami                         = data.aws_ami.amazon_linux2.id
  instance_type               = "t3.small"
  key_name                    = aws_key_pair.traffic_ai[0].key_name
  vpc_security_group_ids      = [aws_security_group.traffic_sg[0].id]
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

    # restart container
    docker rm -f traffic-ai || true

    # If you want EC2 to call SageMaker for predictions:
    SM_ENDPOINT="${aws_sagemaker_endpoint.serverless.name}"

    GMAPS="$(aws secretsmanager get-secret-value --region "$REGION" \
      --secret-id 'traffic-ai/google-map-key' \
      --query SecretString --output text | jq -r '.key')"

    docker run -d --name traffic-ai -p 8080:8080 --restart unless-stopped \
      -e USE_LIVE_APIS=true \
      -e USE_SM=true \
      -e RAW_BUCKET="${aws_s3_bucket.raw.bucket}" \
      -e ATHENA_RESULTS_BUCKET="${aws_s3_bucket.athena_results.bucket}" \
      -e AWS_REGION="$REGION" \
      -e AWS_DEFAULT_REGION="$REGION" \
      -e SM_ENDPOINT="${aws_sagemaker_endpoint.serverless.name}" \
      -e GOOGLE_MAPS_API_KEY="$GMAPS" \
      -e GOOGLE_MAP_KEY="$GMAPS" \
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

#############################
# Elastic IP
#############################
resource "aws_eip" "traffic_ip" {
  count  = var.enable_single_ec2 ? 1 : 0
  domain = "vpc"
}

resource "aws_eip_association" "traffic_assoc" {
  count         = var.enable_single_ec2 ? 1 : 0
  instance_id   = aws_instance.traffic_ec2[0].id
  allocation_id = aws_eip.traffic_ip[0].id
}

#############################
# Outputs
#############################
output "ec2_public_ip" {
  value       = var.enable_single_ec2 ? aws_eip.traffic_ip[0].public_ip : null
  description = "Single-EC2 public IP (null when disabled)"
}
output "ec2_url" {
  value       = var.enable_single_ec2 ? "http://${aws_eip.traffic_ip[0].public_ip}:8080/healthz" : null
  description = "Single-EC2 health URL (null when disabled)"
}

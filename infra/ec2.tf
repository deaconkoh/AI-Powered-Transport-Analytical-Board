# --- Security Group ---
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
    cidr_blocks = ["0.0.0.0/0"] # TIP: replace with your IP in prod
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-sg" }
}

# --- AMI (Amazon Linux 2) ---
data "aws_ami" "amazon_linux2" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# --- Key Pair (uses your local .pub) ---
resource "aws_key_pair" "traffic_ai" {
  key_name   = "traffic-ai-key"
  public_key = file(pathexpand("~/.ssh/traffic-ai-key.pub"))

  lifecycle {
    ignore_changes = [public_key]  # <- prevent forced replacement
  }
}

# --- (Minimal) IAM for ECR pull ---
# Ensure your instance profile has this managed policy attached:
# data "aws_iam_policy" "ecr_ro" { arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly" }
# ...then attach to the role used by aws_iam_instance_profile.ec2_profile

# --- EC2 Instance ---
resource "aws_instance" "traffic_ec2" {
  ami                         = data.aws_ami.amazon_linux2.id
  instance_type               = "t3.micro"
  key_name                    = aws_key_pair.traffic_ai.key_name
  vpc_security_group_ids      = [aws_security_group.traffic_sg.id]
  associate_public_ip_address = true
  root_block_device {
    volume_size = 30       
    volume_type = "gp3"
    iops        = 3000
    throughput  = 125
  }
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name

  # Create new before destroying old, so you don't lose the box during changes
  lifecycle { create_before_destroy = true }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail

    yum update -y
    yum install -y docker awscli
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ec2-user

    REGION="${var.region}"
    REPO_URL="${aws_ecr_repository.app.repository_url}"
    IMAGE_TAG="${var.image_tag}"

    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REPO_URL"
    docker pull "$REPO_URL:$IMAGE_TAG"

    # Clean previous container if re-provisioning
    docker rm -f traffic-ai || true

    # IMPORTANT: app must bind to 0.0.0.0:8080 inside the container
    docker run -d --name traffic-ai -p 8080:8080 --restart unless-stopped "$REPO_URL:$IMAGE_TAG"
    EOF


  tags = { Name = "${var.project_name}-ec2" }

  depends_on = [
    aws_ecr_repository.app
    # plus your IAM attachments that grant ECR read to the instance role
  ]
}

# --- Keep the same public IP across recreates ---
resource "aws_eip" "traffic_ip" {
  domain = "vpc"
}
resource "aws_eip_association" "traffic_assoc" {
  instance_id   = aws_instance.traffic_ec2.id
  allocation_id = aws_eip.traffic_ip.id
}

# --- Outputs ---
output "ec2_public_ip" {
  value = aws_eip.traffic_ip.public_ip # use EIP if created, else aws_instance.traffic_ec2.public_ip
}

output "ec2_url" {
  value = "http://${aws_eip.traffic_ip.public_ip}:8080/healthz" # ensure /healthz exists; otherwise use "/"
}
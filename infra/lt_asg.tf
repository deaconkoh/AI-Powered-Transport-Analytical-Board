resource "aws_launch_template" "api_lt" {
  name_prefix            = "${var.project_name}-lt-"
  image_id               = data.aws_ami.amazon_linux2.id
  instance_type          = "t3.small"
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]

  # you already have an instance profile in iam.tf
  iam_instance_profile { name = aws_iam_instance_profile.ec2_profile.name }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -euxo pipefail
    yum install -y docker awscli jq
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ec2-user || true

    REGION="${var.region}"
    REPO_URL="${aws_ecr_repository.app.repository_url}"
    IMAGE_TAG="${var.image_tag}"

    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REPO_URL"
    docker pull "$REPO_URL:$IMAGE_TAG" || true
    docker rm -f traffic-ai || true

    GMAPS="$(aws secretsmanager get-secret-value --region "$REGION" \
      --secret-id '${var.project_name}/google-map-key' \
      --query SecretString --output text | jq -r '.key')"

    docker run -d --name traffic-ai -p 8080:8080 --restart unless-stopped \
      -e USE_LIVE_APIS=false \
      -e USE_SM=true \
      -e RAW_BUCKET="${aws_s3_bucket.raw.bucket}" \
      -e AWS_REGION="$REGION" \
      -e AWS_DEFAULT_REGION="$REGION" \
      -e SM_ENDPOINT="${aws_sagemaker_endpoint.serverless.name}" \
      -e GOOGLE_MAPS_API_KEY="$GMAPS" \
      "$REPO_URL:$IMAGE_TAG"
  EOF
  )
}

resource "aws_autoscaling_group" "api_asg" {
  name                      = "${var.project_name}-asg"
  max_size                  = 1
  min_size                  = 1
  desired_capacity          = 1
  vpc_zone_identifier       = var.public_subnet_ids
  health_check_type         = "ELB"
  health_check_grace_period = 180

  launch_template {
    id      = aws_launch_template.api_lt.id
    version = "$Latest"
  }
  target_group_arns = [aws_lb_target_group.api_tg.arn]

  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 90
      instance_warmup        = 60
    }
  }

  lifecycle { create_before_destroy = true }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-api"
    propagate_at_launch = true
  }
}

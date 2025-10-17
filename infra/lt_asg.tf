resource "aws_launch_template" "api_lt" {
  name_prefix            = "${var.project_name}-lt-"
  image_id               = data.aws_ami.amazon_linux2.id
  instance_type          = "t3.small"
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]

  # you already have an instance profile in iam.tf
  iam_instance_profile { name = aws_iam_instance_profile.ec2_profile.name }
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 30
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
    }
  }

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

    # Version bump to trigger LT update
    docker system prune -af || true

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
  max_size                  = var.asg_max_size
  min_size                  = var.asg_min_size
  desired_capacity          = var.asg_desired_capacity
  vpc_zone_identifier       = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  health_check_type         = "ELB"
  health_check_grace_period = 300

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

# CPU target tracking (scale out/in slowly)
resource "aws_autoscaling_policy" "cpu_tgt" {
  name                   = "${var.project_name}-cpu-tt"
  autoscaling_group_name = aws_autoscaling_group.api_asg.name
  policy_type            = "TargetTrackingScaling"

  target_tracking_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ASGAverageCPUUtilization"
    }
    target_value = 70 # scale out above ~70% avg CPU
  }

  estimated_instance_warmup = 300
}
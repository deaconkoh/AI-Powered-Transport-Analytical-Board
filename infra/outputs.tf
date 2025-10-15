output "alb_name" {
  value = aws_lb.api_alb.name
}

output "alb_log_bucket" {
  value = aws_s3_bucket.alb_logs.bucket
}

output "asg_name" {
  value = aws_autoscaling_group.api_asg.name
}

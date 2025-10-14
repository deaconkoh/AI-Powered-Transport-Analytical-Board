locals { repo_name = "${var.project_name}-repo" }

resource "aws_ecr_repository" "app" {
  name                 = local.repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "alb_dns_name" {
  value = aws_lb.api_alb.dns_name
}
output "alb_arn" {
  value = aws_lb.api_alb.arn
}

output "alb_target_group_arn" {
  value       = aws_lb_target_group.api_tg.arn
  description = "Target group ARN for health checks/registration"
}

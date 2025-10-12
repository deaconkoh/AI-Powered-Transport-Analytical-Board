locals { repo_name = "${var.project_name}-repo" }

resource "aws_ecr_repository" "app" {
  name                 = local.repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

#############################
# Secrets (Google Maps only on EC2)
#############################

# Create the container/metadata with TF (no value here),
resource "aws_secretsmanager_secret" "google_maps" {
  name        = "${var.project_name}/google-map-key"
  description = "Google Maps API key for ${var.project_name}"
}

output "google_maps_secret_name" {
  value = aws_secretsmanager_secret.google_maps.name
}
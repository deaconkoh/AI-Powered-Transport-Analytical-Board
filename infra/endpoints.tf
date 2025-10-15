resource "aws_vpc_endpoint" "s3" {
  count             = var.enable_private_networking ? 1 : 0
  vpc_id            = var.vpc_id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags = { Name = "${var.project_name}-vpce-s3" }
}

#resource "aws_vpc_endpoint" "secretsmanager" {
#  count              = var.enable_private_networking ? 1 : 0
#  vpc_id             = var.vpc_id
#  service_name       = "com.amazonaws.${var.region}.secretsmanager"
#  vpc_endpoint_type  = "Interface"
#  subnet_ids         = [aws_subnet.private_a.id, aws_subnet.private_b.id]
#  security_group_ids = [aws_security_group.ec2_sg.id]
#  private_dns_enabled = true
#  tags = { Name = "${var.project_name}-vpce-secrets" }
#}

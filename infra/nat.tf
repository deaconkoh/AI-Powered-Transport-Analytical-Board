resource "aws_eip" "nat" {
  domain = "vpc"
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = data.aws_subnet.public_a.id  # NAT sit in a public subnet
  tags = { Name = "${var.project_name}-nat" }
}

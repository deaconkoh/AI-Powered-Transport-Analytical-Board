resource "aws_subnet" "private_a" {
  vpc_id                  = var.vpc_id
  cidr_block              = "172.31.200.0/24" 
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = false
  tags = { Name = "${var.project_name}-private-a" }
}

resource "aws_subnet" "private_b" {
  vpc_id                  = var.vpc_id
  cidr_block              = "172.31.201.0/24"  
  availability_zone       = "us-east-1c"
  map_public_ip_on_launch = false
  tags = { Name = "${var.project_name}-private-b" }
}

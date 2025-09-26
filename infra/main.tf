terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = "us-east-1" }

# Default VPC + subnets
data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter { name = "vpc-id" values = [data.aws_vpc.default.id] }
}

# Latest Amazon Linux 2023
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["137112412989"]
  filter { name = "name" values = ["al2023-ami-*-x86_64"] }
}

# Security Group: open 80 and 8000
resource "aws_security_group" "mvp_sg" {
  description = "Allow HTTP 80 and Flask 8000"
  vpc_id      = data.aws_vpc.default.id

  ingress { from_port = 80   to_port = 80   protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  ingress { from_port = 8000 to_port = 8000 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }

  egress  { from_port = 0 to_port = 0 protocol = "-1" cidr_blocks = ["0.0.0.0/0"] }

  tags = { Name = "routing-mvp-sg" }
}

# IAM role so EC2 can read SSM parameter (SecureString needs KMS decrypt)
resource "aws_iam_role" "ec2_role" {
  name = "routing-mvp-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{ Effect="Allow", Principal={ Service="ec2.amazonaws.com" }, Action="sts:AssumeRole" }]
  })
}

resource "aws_iam_policy" "ec2_ssm_policy" {
  name = "routing-mvp-ec2-ssm"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect  = "Allow"
        Action  = ["ssm:GetParameter"]
        Resource = "*"
      },
      {
        Effect  = "Allow"
        Action  = ["kms:Decrypt"]
        Resource = "*" # later: restrict to the KMS key for SSM, e.g. arn:aws:kms:...:key/...
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_ssm" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_ssm_policy.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "routing-mvp-instance-profile"
  role = aws_iam_role.ec2_role.name
}

# EC2 instance
resource "aws_instance" "web" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = "t3.micro"
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.mvp_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true

  user_data = <<'EOF'
#!/bin/bash
set -xe
dnf update -y
dnf install -y python3 git nginx awscli
python3 -m venv /opt/appenv
source /opt/appenv/bin/activate
pip install --upgrade pip gunicorn

# fetch app code
rm -rf /opt/app
git clone https://github.com/deaconkoh/AI-Powered-Transport-Analytical-Board.git /opt/app

# get Google Maps key from SSM and write .env
GOOGLE_KEY=$(aws ssm get-parameter --name "/routing/google_map_key" --with-decryption --query "Parameter.Value" --output text)
echo "GOOGLE_MAP_KEY=${GOOGLE_KEY}" > /opt/app/.env

# systemd service for gunicorn
cat >/etc/systemd/system/routing.service <<'SYS'
[Unit]
Description=Routing Flask App
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/app
Environment="PATH=/opt/appenv/bin"
ExecStart=/opt/appenv/bin/gunicorn -w 2 -b 0.0.0.0:8000 server:app
Restart=always

[Install]
WantedBy=multi-user.target
SYS

# install deps & start
pip install -r /opt/app/requirements.txt || true
systemctl daemon-reload
systemctl enable routing
systemctl restart routing

# nginx reverse proxy :80 -> :8000
cat >/etc/nginx/conf.d/routing.conf <<'NGX'
server {
  listen 80 default_server;
  server_name _;
  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
}
NGX
rm -f /etc/nginx/conf.d/default.conf || true
systemctl enable nginx
systemctl restart nginx
EOF

  tags = { Name = "routing-phase2" }
}

output "public_ip"   { value = aws_instance.web.public_ip }
output "url_http80"  { value = "http://${aws_instance.web.public_ip}/" }
output "route_post"  { value = "http://${aws_instance.web.public_ip}:8000/route" }

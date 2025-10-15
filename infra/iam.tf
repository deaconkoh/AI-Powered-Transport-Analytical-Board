#############################
# IAM for EC2 -> ECR + Secret (Google only)
#############################

resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ec2.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "raw_s3_read_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.raw_s3_read.arn
}

resource "aws_iam_policy" "raw_s3_read" {
  name = "${var.project_name}-raw-s3-read"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["s3:ListBucket"],
        Resource = "arn:aws:s3:::${aws_s3_bucket.raw.bucket}",
        Condition = {
          StringLike = {
            "s3:prefix" : [
              "raw/lta/carparks/*",
              "raw/lta/carparks/"
            ]
          }
        }
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject"],
        Resource = "arn:aws:s3:::${aws_s3_bucket.raw.bucket}/raw/lta/carparks/*"
      }
    ]
  })
}

# ECR pull
resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Allow EC2 to read ONLY the Google Maps secret
resource "aws_iam_policy" "read_google_secret" {
  name = "${var.project_name}-read-google-secret"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect : "Allow",
      Action : ["secretsmanager:GetSecretValue"],
      Resource : aws_secretsmanager_secret.google_maps.arn
    }]
  })
}
resource "aws_iam_role_policy_attachment" "read_google_secret_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.read_google_secret.arn
}

# Instance profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}
resource "aws_iam_policy" "invoke_sm" {
  name = "${var.project_name}-invoke-sm"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = "sagemaker:InvokeEndpoint",
      Resource = "*" // or the specific endpoint ARN for tighter scope
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_invoke_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.invoke_sm.arn
}

# Give the instance exactly what AmazonSSMManagedInstanceCore provides, as INLINE policy
resource "aws_iam_role_policy" "ssm_core_inline" {
  name = "${var.project_name}-ssm-core-inline"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # SSM core
      {
        Effect = "Allow",
        Action = [
          "ssm:DescribeAssociation",
          "ssm:GetDeployablePatchSnapshotForInstance",
          "ssm:GetDocument",
          "ssm:DescribeDocument",
          "ssm:GetManifest",
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:ListAssociations",
          "ssm:ListInstanceAssociations",
          "ssm:PutInventory",
          "ssm:PutComplianceItems",
          "ssm:PutConfigurePackageResult",
          "ssm:UpdateAssociationStatus",
          "ssm:UpdateInstanceAssociationStatus",
          "ssm:UpdateInstanceInformation"
        ],
        Resource = "*"
      },
      # EC2 messages
      {
        Effect = "Allow",
        Action = [
          "ec2messages:AcknowledgeMessage",
          "ec2messages:DeleteMessage",
          "ec2messages:FailMessage",
          "ec2messages:GetEndpoint",
          "ec2messages:GetMessages",
          "ec2messages:SendReply"
        ],
        Resource = "*"
      },
      # SSM messages (Session Manager channels)
      {
        Effect = "Allow",
        Action = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ],
        Resource = "*"
      },

      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_group" "terraform_admins" {
  name = "terraform-admins"
}

# A managed policy with the exact WAF + Logs perms Terraform needs
resource "aws_iam_policy" "waf_manage" {
  name        = "${var.project_name}-waf-manage"
  description = "Allow Terraform to create/attach WAF and enable WAF logging"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # WebACL CRUD + association
      {
        Effect   = "Allow",
        Action   = [
          "wafv2:CreateWebACL",
          "wafv2:UpdateWebACL",
          "wafv2:DeleteWebACL",
          "wafv2:GetWebACL",
          "wafv2:ListWebACLs",
          "wafv2:AssociateWebACL",
          "wafv2:DisassociateWebACL",
          "wafv2:GetWebACLForResource",
          "wafv2:ListTagsForResource",
          "wafv2:TagResource", 
          "wafv2:UntagResource"
        ],
        Resource = "*"
      },
      # Managed rule sets (needed for AWSManagedRules*)
      {
        Effect   = "Allow",
        Action   = [
          "wafv2:ListAvailableManagedRuleGroups",
          "wafv2:ListAvailableManagedRuleGroupVersions",
          "wafv2:DescribeManagedRuleGroup",
          "wafv2:GetManagedRuleSet",
          "wafv2:ListManagedRuleSets"
        ],
        Resource = "*"
      },
      # WAF logging configuration
      {
        Effect   = "Allow",
        Action   = [
          "wafv2:PutLoggingConfiguration",
          "wafv2:DeleteLoggingConfiguration",
          "wafv2:GetLoggingConfiguration",
          "wafv2:ListLoggingConfigurations"
        ],
        Resource = "*"
      },
      # Create the CW Logs group and resource policy for WAF logging
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:PutRetentionPolicy",
          "logs:DeleteLogGroup",
          "logs:DescribeLogGroups",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DeleteResourcePolicy"
        ],
        Resource = "*"
      },
      # Read-only to discover ALB/TG ARNs
      {
        Effect   = "Allow",
        Action   = [
          "elasticloadbalancing:DescribeLoadBalancers",
          "elasticloadbalancing:DescribeTargetGroups"
        ],
        Resource = "*"
      }
    ]
  })
}

# Attach that managed policy to the group
resource "aws_iam_group_policy_attachment" "terraform_admins_waf_manage" {
  group      = aws_iam_group.terraform_admins.name
  policy_arn = aws_iam_policy.waf_manage.arn
}

# Put terraform user in that group
data "aws_iam_user" "terraform_user" {
  user_name = "terraform-user"
}

resource "aws_iam_user_group_membership" "terraform_user_in_group" {
  user = data.aws_iam_user.terraform_user.user_name
  groups = [aws_iam_group.terraform_admins.name]
}
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

# Glue execution role
resource "aws_iam_role" "glue_role" {
  name = "${var.project_name}-glue-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "glue.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

# Basic Glue service permissions
resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

# Allow Glue to read/write your raw bucket (Bronze/Silver/Gold)
resource "aws_iam_policy" "glue_s3_access" {
  name = "${var.project_name}-glue-s3-access"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      Resource = [
        "arn:aws:s3:::${local.raw_bucket_name}",
        "arn:aws:s3:::${local.raw_bucket_name}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "glue_s3_access_attach" {
  role       = aws_iam_role.glue_role.name
  policy_arn = aws_iam_policy.glue_s3_access.arn
}

# ==========================================
# Permissions for MLOps (Glue Batch Job)
# ==========================================

# Allow Glue to read from the Models bucket and SSM
resource "aws_iam_policy" "glue_mlops_policy" {
  name = "${var.project_name}-glue-mlops-access"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # 1. Read Model Artifacts & Scripts from Models Bucket
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject", 
          "s3:ListBucket"
        ],
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      # 2. Read the SSM Parameter to find the current model
      {
        Effect = "Allow",
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ],
        Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/traffic/hybrid/*"
      }
    ]
  })
}

# Attach it to your existing Glue Role
resource "aws_iam_role_policy_attachment" "glue_mlops_attach" {
  role       = aws_iam_role.glue_role.name
  policy_arn = aws_iam_policy.glue_mlops_policy.arn
}

# Policy for EC2 to access Athena
resource "aws_iam_policy" "ec2_athena_access" {
  name = "${var.project_name}-ec2-athena-access"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "athena:StartQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "glue:GetTable",
          "glue:GetPartitions",
          "glue:GetDatabase"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = ["s3:Get*", "s3:List*", "s3:PutObject"],
        Resource = [
          aws_s3_bucket.athena_results.arn,
          "${aws_s3_bucket.athena_results.arn}/*",
          aws_s3_bucket.raw.arn,
          "${aws_s3_bucket.raw.arn}/*"
        ]
      }
    ]
  })
}

# Attach to your existing EC2 Role
resource "aws_iam_role_policy_attachment" "ec2_athena_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_athena_access.arn
}
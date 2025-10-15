data "aws_subnet" "public_a" { id = var.public_subnet_ids[0] }
data "aws_subnet" "public_b" { id = var.public_subnet_ids[1] }


# Pick TWO subnets you want to become "private" later (reuse existing).
# data "aws_subnet" "private_a" { id = "subnet-06b75c71fd2dc0e88" } # us-east-1e
# data "aws_subnet" "private_b" { id = "subnet-07f5854167bd5adf2" } # us-east-1d

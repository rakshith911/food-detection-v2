# Temporary EC2 instance to push Docker image to ECR
# This creates a small instance with enough disk space to load the 3.12GB image

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security group for temporary EC2 instance
resource "aws_security_group" "docker_push_temp" {
  name_prefix = "docker-push-temp-"
  description = "Temporary SG for Docker push EC2 instance"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-docker-push-temp"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# IAM role for EC2 instance
resource "aws_iam_role" "docker_push_temp" {
  name_prefix = "docker-push-temp-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-${var.environment}-docker-push-temp"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Attach necessary policies
resource "aws_iam_role_policy" "docker_push_temp" {
  name_prefix = "docker-push-temp-"
  role        = aws_iam_role.docker_push_temp.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.videos.arn}",
          "${aws_s3_bucket.videos.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = "${aws_kms_key.main.arn}"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "docker_push_temp" {
  name_prefix = "docker-push-temp-"
  role        = aws_iam_role.docker_push_temp.name
}

# EC2 instance - minimal setup, we'll SSH in and run commands manually
resource "aws_instance" "docker_push_temp" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.public_1.id
  vpc_security_group_ids = [aws_security_group.docker_push_temp.id]
  iam_instance_profile   = aws_iam_instance_profile.docker_push_temp.name

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
  }

  # Download and execute build script from GitHub
  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -x
    exec > >(tee /var/log/user-data.log)
    exec 2>&1

    # Install Docker and git
    yum update -y
    yum install -y docker git
    systemctl start docker
    systemctl enable docker
    usermod -a -G docker ec2-user

    # Build and push Docker image directly
    REGION="us-east-1"
    REPO_NAME="food-detection-v2-worker"
    ACCOUNT_ID=\$(aws sts get-caller-identity --query Account --output text)
    ECR_REPO="\$ACCOUNT_ID.dkr.ecr.\$REGION.amazonaws.com/\$REPO_NAME"

    # Clone repo as ec2-user
    cd /home/ec2-user
    sudo -u ec2-user git clone https://github.com/leolorence12345/food-detection.git
    cd food-detection/FoodAI/nutrition-video-analysis/terraform/docker

    # Login to ECR as ec2-user
    sudo -u ec2-user aws ecr get-login-password --region \$REGION | docker login --username AWS --password-stdin \$ECR_REPO

    # Build
    sudo -u ec2-user docker build -t nutrition-api:latest .

    # Tag
    sudo -u ec2-user docker tag nutrition-api:latest \$ECR_REPO:latest
    sudo -u ec2-user docker tag nutrition-api:latest \$ECR_REPO:\$(date +%Y%m%d-%H%M%S)

    # Push
    sudo -u ec2-user docker push \$ECR_REPO:latest
    sudo -u ec2-user docker push \$ECR_REPO:\$(date +%Y%m%d-%H%M%S)

    # Create success marker
    echo "Build completed at \$(date)" | aws s3 cp - s3://nutrition-video-analysis-dev-videos-dbenpoj2/build-complete.txt
  EOF
  )

  tags = {
    Name        = "${var.project_name}-${var.environment}-docker-push-temp"
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "Temporary instance for Docker push to ECR"
  }
}

output "docker_push_instance_id" {
  value       = aws_instance.docker_push_temp.id
  description = "EC2 instance ID for Docker push"
}

output "docker_push_log_command" {
  value       = "aws ec2 get-console-output --instance-id ${aws_instance.docker_push_temp.id} --region us-east-1 --output text"
  description = "Command to view the Docker push progress"
}

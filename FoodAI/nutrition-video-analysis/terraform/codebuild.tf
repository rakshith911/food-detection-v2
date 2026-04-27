# CodeBuild project for building and pushing Docker images to ECR

# IAM role for CodeBuild
resource "aws_iam_role" "codebuild" {
  name_prefix = "codebuild-docker-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-${var.environment}-codebuild-role"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# IAM policy for CodeBuild
resource "aws_iam_role_policy" "codebuild" {
  name_prefix = "codebuild-docker-"
  role        = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
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
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.videos.arn}",
          "${aws_s3_bucket.videos.arn}/*"
        ]
      }
    ]
  })
}

# CodeBuild project
resource "aws_codebuild_project" "docker_build" {
  name          = "${var.project_name}-${var.environment}-docker-build"
  description   = "Build and push Docker image to ECR"
  build_timeout = 60
  service_role  = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_LARGE"
    image                       = "aws/codebuild/standard:7.0"
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = true

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = var.aws_region
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = data.aws_caller_identity.current.account_id
    }

    environment_variable {
      name  = "ECR_REPO_NAME"
      value = aws_ecr_repository.video_processor.name
    }

    environment_variable {
      name  = "ECS_CLUSTER"
      value = "food-detection-v2-cluster"
    }

    environment_variable {
      name  = "ECS_SERVICE"
      value = "food-detection-v2-worker"
    }

    environment_variable {
      name  = "ECS_TASK_FAMILY"
      value = "food-detection-v2-worker"
    }
  }

  source {
    type            = "GITHUB"
    location        = "https://github.com/rakshith911/food-detection-v2.git"
    git_clone_depth = 1
    buildspec       = "FoodAI/nutrition-video-analysis/terraform/docker/buildspec.yml"
  }

  logs_config {
    cloudwatch_logs {
      group_name  = "/aws/codebuild/${var.project_name}-${var.environment}-docker-build"
      stream_name = "build"
    }
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-docker-build"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Output the CodeBuild project name
output "codebuild_project_name" {
  value       = aws_codebuild_project.docker_build.name
  description = "CodeBuild project name for building Docker images"
}

output "codebuild_start_command" {
  value       = "aws codebuild start-build --project-name ${aws_codebuild_project.docker_build.name} --region ${var.aws_region}"
  description = "Command to start a CodeBuild build"
}

output "codebuild_logs_command" {
  value       = "aws logs tail /aws/codebuild/${aws_codebuild_project.docker_build.name} --follow --region ${var.aws_region}"
  description = "Command to view CodeBuild logs"
}

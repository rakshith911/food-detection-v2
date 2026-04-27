# =============================================================================
# VARIABLES
# =============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "food-detection-v2"
}

variable "environment" {
  description = "Environment"
  type        = string
  default     = "dev"
}

variable "gemini_api_key" {
  description = "Gemini API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "notification_email" {
  description = "Email for notifications"
  type        = string
  default     = "admin@example.com"
}

# GPU Configuration
variable "use_gpu" {
  description = "Enable GPU support (requires EC2 launch type)"
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU tasks (g4dn.xlarge, g5.xlarge, etc.)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_instance_ami" {
  description = "AMI ID for GPU instances (Deep Learning AMI or Amazon Linux with GPU drivers)"
  type        = string
  default     = ""  # Will use data source to find latest Deep Learning AMI if empty
}

variable "gpu_min_capacity" {
  description = "Minimum number of GPU instances"
  type        = number
  default     = 0
}

variable "gpu_max_capacity" {
  description = "Maximum number of GPU instances"
  type        = number
  default     = 5
}

variable "device_type" {
  description = "Device type for processing (cpu or cuda)"
  type        = string
  default     = "cpu"
  validation {
    condition     = contains(["cpu", "cuda"], var.device_type)
    error_message = "Device type must be either 'cpu' or 'cuda'."
  }
}

variable "docker_image_tag" {
  description = "Docker image tag to use (latest, gpu-test, etc.)"
  type        = string
  default     = "latest"
}

variable "gpu_key_pair_name" {
  description = "EC2 Key Pair name for GPU instances (optional, for SSH access)"
  type        = string
  default     = ""
}

# User Data S3 Bucket (for per-user backup/restore)
variable "user_data_bucket" {
  description = "S3 bucket name for per-user data backup (UKcal folders)"
  type        = string
  default     = "ukcal-user-uploads"
}

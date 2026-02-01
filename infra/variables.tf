variable "project_name" {
  type        = string
  description = "Base name for ECS/ECR resources"
  default     = "customer-churn"
}

variable "aws_region" {
  type        = string
  description = "AWS region to deploy into"
  default     = "us-east-1"
}

variable "container_port" {
  type        = number
  description = "Container port exposed by the app"
  default     = 5001
}

variable "desired_count" {
  type        = number
  description = "Number of ECS tasks"
  default     = 1
}

variable "cpu" {
  type        = number
  description = "Fargate CPU units"
  default     = 256
}

variable "memory" {
  type        = number
  description = "Fargate memory (MiB)"
  default     = 512
}

variable "image_tag" {
  type        = string
  description = "Docker image tag to deploy"
  default     = "latest"
}

variable "health_check_path" {
  type        = string
  description = "ALB target group health check path"
  default     = "/health"
}

variable "log_retention_days" {
  type        = number
  description = "CloudWatch log retention in days"
  default     = 14
}
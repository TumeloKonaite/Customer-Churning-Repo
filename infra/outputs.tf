output "alb_url" {
  description = "Public URL of the load balancer"
  value       = "http://${aws_lb.app.dns_name}"
}

output "ecr_repo_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "ecr_repo_name" {
  description = "ECR repository name"
  value       = aws_ecr_repository.app.name
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.app.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

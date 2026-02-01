# Infrastructure (Terraform)

This folder provisions a minimal ECS Fargate stack using the **default VPC**.
It is intentionally small for a portfolio demo, but still production-credible.

## What it creates
- ECR repository
- ECS cluster + task definition + service (Fargate)
- Application Load Balancer + target group + HTTP listener
- CloudWatch log group
- Security groups

## Prereqs
- Terraform >= 1.5
- AWS credentials with permissions for ECR, ECS, ELBv2, CloudWatch Logs, IAM

## Quick start
```bash
cd infra
terraform init
terraform plan -out tfplan
terraform apply tfplan
```

## Outputs
- `alb_url`
- `ecr_repo_url`
- `ecr_repo_name`
- `ecs_cluster_name`
- `ecs_service_name`

## Deploy flow
The ECS task definition uses the image tag from `var.image_tag` (default `latest`).
CI/CD pushes a new `latest` image and forces a new deployment so ECS pulls it.

## Clean up
```bash
terraform destroy
```

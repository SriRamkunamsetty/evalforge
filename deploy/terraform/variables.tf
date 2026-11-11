variable "namespace" {
  type        = string
  default     = "evalforge"
  description = "Target Kubernetes namespace for EvalForge"
}

variable "api_key" {
  type        = string
  sensitive   = true
  description = "Production API Key for EvalForge write operations"
}

variable "secret_key" {
  type        = string
  sensitive   = true
  description = "Flask session secret signing key"
}

variable "domain_name" {
  type        = string
  default     = "evalforge.example.com"
  description = "Public domain name mapped to Ingress"
}

variable "image_repository" {
  type        = string
  default     = "ghcr.io/sriramkunamsetty/evalforge"
  description = "Container image repository"
}

variable "image_tag" {
  type        = string
  default     = "0.1.0"
  description = "Container image release tag"
}

variable "replica_count" {
  type        = number
  default     = 2
  description = "Base replica count"
}

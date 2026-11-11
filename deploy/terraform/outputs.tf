output "namespace" {
  value       = kubernetes_namespace.evalforge.metadata[0].name
  description = "Created Kubernetes namespace"
}

output "helm_release_status" {
  value       = helm_release.evalforge.status
  description = "Helm release status"
}

output "service_url" {
  value       = "https://${var.domain_name}"
  description = "EvalForge public service URL"
}

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.13.0"
    }
  }
}

resource "kubernetes_namespace" "evalforge" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "evalforge"
      "pod-security.kubernetes.io/enforce" = "restricted"
    }
  }
}

resource "kubernetes_secret" "evalforge_secrets" {
  metadata {
    name      = "evalforge-secrets"
    namespace = kubernetes_namespace.evalforge.metadata[0].name
  }

  data = {
    API_KEY    = var.api_key
    SECRET_KEY = var.secret_key
  }

  type = "Opaque"
}

resource "helm_release" "evalforge" {
  name       = "evalforge"
  repository = "deploy/helm/evalforge"
  chart      = "evalforge"
  namespace  = kubernetes_namespace.evalforge.metadata[0].name

  values = [
    yamlencode({
      image = {
        repository = var.image_repository
        tag        = var.image_tag
      }
      replicaCount = var.replica_count
      ingress = {
        enabled = true
        hosts = [{
          host = var.domain_name
          paths = [{
            path     = "/"
            pathType = "Prefix"
          }]
        }]
      }
    })
  ]

  depends_on = [
    kubernetes_secret.evalforge_secrets
  ]
}

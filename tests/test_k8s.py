"""Manifests Kubernetes : règles de sécurité et de disponibilité vérifiées automatiquement."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
K8S = Path(__file__).resolve().parents[1] / "k8s"


def load(name):
    return yaml.safe_load((K8S / name).read_text())


def test_deployment_hardening():
    spec = load("deployment.yaml")["spec"]["template"]["spec"]
    c = spec["containers"][0]
    assert spec["automountServiceAccountToken"] is False
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    sc = c["securityContext"]
    assert sc["readOnlyRootFilesystem"] and not sc["allowPrivilegeEscalation"] and sc["capabilities"]["drop"] == ["ALL"]
    assert c["resources"]["requests"] and c["resources"]["limits"]
    assert c["readinessProbe"]["httpGet"]["path"] == "/health" and c["livenessProbe"]
    assert not c["image"].endswith(":latest") and ":" in c["image"]
    assert c["env"][0]["valueFrom"]["secretKeyRef"]["name"] == "catdog-api"


def test_namespace_restricted_and_secret_not_deployed():
    assert load("namespace.yaml")["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"
    assert "secret.example.yaml" not in load("kustomization.yaml")["resources"]


def test_network_policy_denies_egress():
    np_ = load("networkpolicy.yaml")["spec"]
    assert "Egress" in np_["policyTypes"] and np_["egress"] == []

from fastapi.testclient import TestClient

from app.main import app


def test_liveness_describes_the_scaffold():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "service": "ml-service",
        "status": "ok",
        "stage": "scaffold",
    }


def test_readiness_discloses_its_scope():
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["scope"] == "application_startup_only"

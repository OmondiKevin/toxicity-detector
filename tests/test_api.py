"""
Test API endpoints.
"""
from fastapi.testclient import TestClient


def test_api_import():
    """Test that API can be imported."""
    from src.api import app

    assert app is not None


def test_health_endpoint():
    """Test health check endpoint."""
    from src.api import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint():
    """Test version endpoint."""
    from src.api import app

    client = TestClient(app)
    response = client.get("/version")

    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data

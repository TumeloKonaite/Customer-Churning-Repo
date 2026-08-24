import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


def test_configured_frontend_origin_can_call_api(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FRONTEND_ALLOWED_ORIGINS", "https://churn.example.com")
    client = TestClient(create_app())

    response = client.options(
        "/health",
        headers={
            "Origin": "https://churn.example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Accept",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://churn.example.com"
    assert "GET" in response.headers["access-control-allow-methods"]


def test_unlisted_origin_is_not_allowed(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FRONTEND_ALLOWED_ORIGINS", "https://churn.example.com")
    client = TestClient(create_app())

    response = client.options(
        "/health",
        headers={"Origin": "https://other.example.com", "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_wildcard_origin_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FRONTEND_ALLOWED_ORIGINS", "*")

    with pytest.raises(ValueError, match="exact origins"):
        create_app()

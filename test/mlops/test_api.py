from fastapi.testclient import TestClient

from trading_chaos_api.main import create_app


async def healthy_database() -> str:
    return "16.4.0"


async def unavailable_database() -> str:
    raise OSError("connection refused")


def test_healthz_does_not_depend_on_database():
    client = TestClient(create_app(unavailable_database))
    assert client.get("/healthz").json() == {"status": "ok"}


def test_version():
    client = TestClient(create_app(healthy_database))
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


def test_dependency_health():
    client = TestClient(create_app(healthy_database))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["components"]["database"]["version"] == "16.4.0"
    assert response.json()["components"]["database"]["response_ms"] >= 0


def test_dependency_failure():
    client = TestClient(create_app(unavailable_database))
    response = client.get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["components"]["database"]["status"] == "error"

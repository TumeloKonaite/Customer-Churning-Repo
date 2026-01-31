import requests
import pytest
import time
import docker
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCKER_TESTS") != "1",
    reason="Docker integration tests are opt-in. Set RUN_DOCKER_TESTS=1 to run.",
)

@pytest.fixture
def docker_container():
    client = docker.from_env()
    container = client.containers.run(
        "churn-predictor:latest",
        ports={'5000/tcp': 5000},
        detach=True
    )
    time.sleep(2)  # Wait for container to start
    yield container
    container.stop()
    container.remove()

def test_docker_health_check(docker_container):
    response = requests.get("http://localhost:5000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_docker_prediction_endpoint(docker_container):
    test_data = {
        "age": 35,
        "tenure": 5,
        "balance": 50000,
        "products": 2
    }
    response = requests.post("http://localhost:5000/api/predict", json=test_data)
    assert response.status_code == 200
    assert "p_churn" in response.json()
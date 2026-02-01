import os
import time
import docker
import pytest
import requests
from docker.errors import DockerException


def _docker_available() -> bool:
    """Check if Docker is available in the current environment."""
    return os.path.exists("/var/run/docker.sock")


def wait_for_health(base_url: str, timeout_s: int = 60) -> None:
    """Wait until /health returns JSON status=healthy."""
    start = time.time()
    last_error = None

    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "healthy":
                    return
        except Exception as e:
            last_error = e
        time.sleep(1)

    raise RuntimeError(
        f"Service did not become healthy within {timeout_s}s. Last error: {last_error}"
    )


@pytest.fixture(scope="module")
def docker_client():
    """Fixture to provide Docker client with proper environment checks."""
    if os.getenv("RUN_DOCKER_TESTS") != "1":
        pytest.skip("Docker tests disabled (set RUN_DOCKER_TESTS=1)")

    if not _docker_available():
        pytest.skip("Docker not available in this environment (e.g., SageMaker Studio).")

    try:
        return docker.from_env()
    except DockerException as e:
        pytest.skip(f"Docker daemon not reachable: {e}")


@pytest.fixture(scope="module")
def docker_container(docker_client):
    """Build and run the container; expose container:5001 on a random host port."""
    image_tag = "churn-predictor:test"

    # Build image
    try:
        docker_client.images.build(path=".", tag=image_tag)
    except docker.errors.BuildError as e:
        pytest.fail(f"Failed to build Docker image: {e}")

    # Run container (publish to random host port)
    try:
        container = docker_client.containers.run(
            image_tag,
            ports={"5001/tcp": None},  # random host port
            detach=True,
            remove=True,
        )
    except docker.errors.APIError as e:
        pytest.fail(f"Failed to start container: {e}")

    try:
        # Discover mapped host port
        container.reload()
        port_info = container.attrs["NetworkSettings"]["Ports"]["5001/tcp"]
        host_port = int(port_info[0]["HostPort"])
        base_url = f"http://localhost:{host_port}"

        wait_for_health(base_url, timeout_s=60)

        # yield both container and base_url for tests
        yield container, base_url

    finally:
        try:
            container.stop(timeout=2)
        except Exception:
            pass  # best effort cleanup


def test_docker_health_check(docker_container):
    """Test the health check endpoint."""
    _, base_url = docker_container
    response = requests.get(f"{base_url}/health", timeout=5)

    assert response.status_code == 200
    assert "application/json" in response.headers.get("Content-Type", "")

    data = response.json()
    assert data["status"] == "healthy"
    assert data.get("model_loaded") is True


def test_docker_prediction_endpoint(docker_container):
    """Test the prediction endpoint with valid data."""
    _, base_url = docker_container

    test_data = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
    }

    response = requests.post(
        f"{base_url}/api/predict",
        json=test_data,
        timeout=15,
    )

    assert response.status_code == 200
    assert "application/json" in response.headers.get("Content-Type", "")

    body = response.json()
    assert body["status"] == "success"
    assert "p_churn" in body
    assert body["p_churn"] is not None
    assert 0.0 <= float(body["p_churn"]) <= 1.0

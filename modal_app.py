"""Modal deployment entrypoint for the customer churn FastAPI application."""

from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "customer-churn-backend"
app = modal.App(APP_NAME)

# The verified package is produced locally before deploy and copied with the app.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path="/app",
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".github",
            ".github/**",
            ".venv",
            ".venv/**",
            ".env",
            ".env.*",
            ".modal.toml",
            "**/__pycache__",
            "**/__pycache__/**",
            ".pytest_cache",
            ".pytest_cache/**",
            ".mypy_cache",
            ".mypy_cache/**",
            ".ruff_cache",
            ".ruff_cache/**",
            "tests",
            "tests/**",
            "notebooks",
            "notebooks/**",
            "dataset",
            "dataset/**",
            "logs",
            "logs/**",
            "artifacts",
            "artifacts/**",
            "*.log",
        ],
    )
    .workdir("/app")
)

runtime_secrets = [modal.Secret.from_name("customer-churn-production")]


@app.function(
    image=image,
    secrets=runtime_secrets,
    timeout=600,
    min_containers=0,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    """Return the FastAPI ASGI application for Modal to serve."""
    from src.config import DatabaseSettings
    from src.database import check_connectivity
    from src.mlops.deployment import validate_production_startup

    # These checks run once while the container starts. They never contact DagsHub.
    validate_production_startup("/app/build/model")
    check_connectivity(DatabaseSettings())
    from application import app as fastapi_application

    return fastapi_application

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


def _execute_monitoring(scheduled_for: str | None = None):
    """Build monitoring-only dependencies inside a non-request Modal container."""
    from datetime import datetime

    from src.config import DatabaseSettings, MonitoringSettings
    from src.database import create_database_engine
    from src.monitoring.__main__ import _store
    from src.monitoring.job import MonitoringJob
    from src.monitoring.repository import MonitoringRepository

    settings = MonitoringSettings()
    engine = create_database_engine(DatabaseSettings())
    try:
        as_of = (
            datetime.fromisoformat(scheduled_for.replace("Z", "+00:00"))
            if scheduled_for
            else None
        )
        return MonitoringJob(MonitoringRepository(engine), _store(settings)).run(
            environment=settings.environment.value,
            model_version_id=settings.model_version_id,
            scheduled_for=as_of,
        )
    finally:
        engine.dispose()


monitoring_retries = modal.Retries(
    max_retries=3,
    backoff_coefficient=2.0,
    initial_delay=5.0,
    max_delay=60.0,
)


@app.function(
    image=image,
    secrets=runtime_secrets,
    schedule=modal.Cron("15 */6 * * *", timezone="UTC"),
    retries=monitoring_retries,
    timeout=1800,
)
def scheduled_monitoring():
    """Run on the policy-v1 cadence, outside all FastAPI request handling."""
    return _execute_monitoring()


@app.function(
    image=image,
    secrets=runtime_secrets,
    retries=monitoring_retries,
    timeout=1800,
)
def run_monitoring(scheduled_for: str | None = None):
    """Manual operations/debug entrypoint; scheduled_for is an optional ISO-8601 time."""
    return _execute_monitoring(scheduled_for)

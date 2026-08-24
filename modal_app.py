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


def _execute_label_materialization(as_of: str | None = None):
    """Build label-only dependencies inside the scheduled worker."""
    from datetime import datetime, timedelta, timezone

    from src.config import DatabaseSettings, OutcomeMonitoringSettings
    from src.database import create_database_engine
    from src.monitoring.label_repository import LabelRepository
    from src.monitoring.labels import LabelMaterializationJob

    settings = OutcomeMonitoringSettings()
    engine = create_database_engine(DatabaseSettings())
    try:
        if as_of:
            observed_at = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        else:
            current = datetime.now(timezone.utc)
            observed_at = current.replace(hour=2, minute=45, second=0, microsecond=0)
            if observed_at > current:
                observed_at -= timedelta(days=1)
        return LabelMaterializationJob(
            LabelRepository(engine),
            required_sources=settings.required_sources,
            horizon_days=settings.horizon_days,
            grace_period_days=settings.grace_period_days,
            label_contract_version=settings.label_contract_version,
        ).run(
            environment=settings.environment.value,
            is_simulated=False,
            as_of=observed_at,
        )
    finally:
        engine.dispose()


@app.function(
    image=image,
    secrets=runtime_secrets,
    schedule=modal.Cron("45 2 * * *", timezone="UTC"),
    retries=monitoring_retries,
    timeout=1800,
)
def scheduled_label_materialization():
    """Daily idempotent attribution, corrections, and matured negatives."""
    return _execute_label_materialization()


@app.function(
    image=image,
    secrets=runtime_secrets,
    retries=monitoring_retries,
    timeout=1800,
)
def run_label_materialization(as_of: str | None = None):
    """Manual label-materialization entrypoint with an optional UTC snapshot."""
    return _execute_label_materialization(as_of)


def _execute_performance(as_of: str | None = None):
    from datetime import datetime, timedelta, timezone

    from src.config import DatabaseSettings, MonitoringSettings, OutcomeMonitoringSettings
    from src.database import create_database_engine
    from src.monitoring.__main__ import _store
    from src.monitoring.label_repository import LabelRepository
    from src.monitoring.labels import LabelMaterializationJob
    from src.monitoring.performance_job import PerformanceJob, PerformanceRepository

    settings = OutcomeMonitoringSettings()
    artifact_settings = MonitoringSettings()
    if as_of:
        evaluated_at = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    else:
        current = datetime.now(timezone.utc)
        days_since_monday = current.weekday()
        evaluated_at = (
            current - timedelta(days=days_since_monday)
        ).replace(hour=3, minute=30, second=0, microsecond=0)
        if evaluated_at > current:
            evaluated_at -= timedelta(days=7)
    engine = create_database_engine(DatabaseSettings())
    try:
        label_summary = LabelMaterializationJob(
            LabelRepository(engine),
            required_sources=settings.required_sources,
            horizon_days=settings.horizon_days,
            grace_period_days=settings.grace_period_days,
            label_contract_version=settings.label_contract_version,
        ).run(environment=settings.environment.value, as_of=evaluated_at)
        cohort_end = evaluated_at - timedelta(
            days=settings.horizon_days + settings.grace_period_days
        )
        cohort_start = cohort_end - timedelta(days=settings.performance_cohort_days)
        return PerformanceJob(
            PerformanceRepository(engine), _store(artifact_settings)
        ).run(
            cohort_start=cohort_start,
            cohort_end=cohort_end,
            horizon_days=settings.horizon_days,
            grace_period_days=settings.grace_period_days,
            outcome_watermark=label_summary["outcome_watermark"],
            label_contract_version=settings.label_contract_version,
            model_version_id=settings.model_version_id,
            deployment_ids=settings.deployment_ids,
            policy_version=settings.policy_version,
            classification_threshold=settings.classification_threshold,
            minimum_privacy_size=settings.minimum_privacy_size,
            label_revision_watermark=label_summary["label_revision_watermark"],
            evaluated_at=evaluated_at,
        )
    finally:
        engine.dispose()


@app.function(
    image=image,
    secrets=runtime_secrets,
    schedule=modal.Cron("30 3 * * 1", timezone="UTC"),
    retries=monitoring_retries,
    timeout=1800,
)
def scheduled_performance_monitoring():
    """Publish a weekly matured-cohort production performance report."""
    return _execute_performance()


@app.function(
    image=image,
    secrets=runtime_secrets,
    retries=monitoring_retries,
    timeout=1800,
)
def run_performance_monitoring(as_of: str | None = None):
    return _execute_performance(as_of)

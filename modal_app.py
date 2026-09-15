"""Modal deployment manifest for the customer churn application."""

import modal

from deployment.modal_resources import build_modal_images

APP_NAME = "customer-churn-backend"
app = modal.App(APP_NAME)
image, arize_export_image = build_modal_images()
runtime_secrets = [modal.Secret.from_name("customer-churn-production")]
arize_secret = modal.Secret.from_name("customer-churn-arize")


# Public HTTP entrypoint. Modal runs the FastAPI ASGI application in the main
# image and allows each warm container to handle up to ten concurrent requests.
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
    from src.workers.api import create_production_api

    return create_production_api()


monitoring_retries = modal.Retries(
    max_retries=3,
    backoff_coefficient=2.0,
    initial_delay=5.0,
    max_delay=60.0,
)


def _execute_arize_export():
    from src.workers.arize_export import execute_arize_export

    return execute_arize_export()


# Scheduled background function. Modal invokes it five minutes after every hour
# to deliver one bounded batch from the transactional outbox to Arize.
@app.function(
    image=arize_export_image,
    secrets=[*runtime_secrets, arize_secret],
    schedule=modal.Cron("5 * * * *", timezone="UTC"),
    retries=monitoring_retries,
    timeout=900,
)
def scheduled_arize_export():
    return _execute_arize_export()


# Manual operations function. It runs the same Arize exporter without a schedule
# so operators can trigger delivery during validation, recovery, or backfills.
@app.function(
    image=arize_export_image,
    secrets=[*runtime_secrets, arize_secret],
    retries=monitoring_retries,
    timeout=900,
)
def run_arize_export():
    return _execute_arize_export()


def _execute_label_materialization(as_of: str | None = None):
    from src.workers.label_materialization import execute_label_materialization

    return execute_label_materialization(as_of)


# Scheduled background function. Modal invokes it daily at 02:45 UTC to resolve
# matured outcomes into actual labels that the Arize exporter can subsequently send.
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


# Manual operations function. It permits an operator-supplied UTC snapshot for
# deterministic label replay and correction processing outside the daily schedule.
@app.function(
    image=image,
    secrets=runtime_secrets,
    retries=monitoring_retries,
    timeout=1800,
)
def run_label_materialization(as_of: str | None = None):
    """Manual label-materialization entrypoint with an optional UTC snapshot."""
    return _execute_label_materialization(as_of)

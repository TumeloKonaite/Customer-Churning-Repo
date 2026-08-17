"""Modal deployment entrypoint for the customer churn Flask application."""

import os
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "customer-churn-backend"
SENDGRID_SECRET_ENV = "MODAL_SENDGRID_SECRET_NAME"

app = modal.App(APP_NAME)

# copy=True is required because training runs in a later image-build layer.
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
            "logs",
            "logs/**",
            "artifacts",
            "artifacts/**",
            "infra/.terraform",
            "infra/.terraform/**",
            "infra/*.tfstate",
            "infra/*.tfstate.*",
            "*.log",
        ],
    )
    .workdir("/app")
    .run_commands("python -m src.train")
)


def _function_secrets() -> list[modal.Secret]:
    """Attach SendGrid only when a deploy explicitly opts into real email."""
    secret_name = os.getenv(SENDGRID_SECRET_ENV, "").strip()
    if not secret_name:
        return []

    return [
        modal.Secret.from_name(
            secret_name,
            required_keys=["SENDGRID_API_KEY", "SENDGRID_VERIFIED_SENDER"],
        )
    ]


@app.function(
    image=image,
    secrets=_function_secrets(),
    timeout=600,
    min_containers=0,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
@modal.wsgi_app()
def flask_app():
    """Return the existing Flask WSGI application for Modal to serve."""
    from application import application

    return application

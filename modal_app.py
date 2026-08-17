"""Modal deployment entrypoint for the customer churn Flask application."""

from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "customer-churn-backend"
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
            "*.log",
        ],
    )
    .workdir("/app")
    .run_commands("python -m src.train")
)


@app.function(
    image=image,
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

"""Reusable Modal image definitions for the application deployment manifest."""

from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_IMAGE_IGNORE = [
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
]


def build_modal_images() -> tuple[modal.Image, modal.Image]:
    """Build the API/label image and the smaller Arize export image."""
    application_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
        .add_local_dir(
            str(PROJECT_ROOT),
            remote_path="/app",
            copy=True,
            ignore=APPLICATION_IMAGE_IGNORE,
        )
        .workdir("/app")
    )
    arize_export_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "arize==8.51.0",
            "pandas",
            "psycopg[binary]>=3.2,<4",
            "pydantic-settings>=2.10,<3",
            "sqlalchemy>=2.0,<3",
        )
        .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/app/src", copy=True)
        .workdir("/app")
    )
    return application_image, arize_export_image

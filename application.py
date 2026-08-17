"""Minimal ASGI and local-development entrypoint."""

import os

from src.api.app import create_app


app = create_app()
# Preserve the historical module-level name used by deployment/import callers.
application = app


def run_app() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "5001"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "0") == "1"
    uvicorn.run("application:app", host="0.0.0.0", port=port, reload=reload_enabled)


if __name__ == "__main__":
    run_app()

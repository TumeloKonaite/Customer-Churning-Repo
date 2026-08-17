PYTHON ?= python
PORT ?= 5001

ifeq ($(OS),Windows_NT)
SMOKE_CMD = powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke.ps1
else
SMOKE_CMD = sh scripts/smoke.sh
endif

.PHONY: run train test smoke clean install reqs modal-serve modal-deploy ci setup all

# Development commands
install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn application:app --host 0.0.0.0 --port $(PORT)

train:
	$(PYTHON) -m src.train

test:
	$(PYTHON) -m pytest

reqs:
	uv lock
	uv export --no-dev --no-emit-project --no-annotate --no-header -o requirements.txt

smoke:
	$(SMOKE_CMD)

# Modal deployment commands
modal-serve:
	modal serve modal_app.py

modal-deploy:
	modal deploy modal_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +

# CI/CD helpers
ci: install test

# All-in-one local setup
setup: clean install train test

# Default target
all: setup

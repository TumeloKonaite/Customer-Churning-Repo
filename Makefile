PYTHON ?= python
IMAGE_NAME = churn-predictor
PORT = 5001

ifeq ($(OS),Windows_NT)
SMOKE_CMD = powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke.ps1
else
SMOKE_CMD = sh scripts/smoke.sh
endif

.PHONY: run train test smoke clean docker-* install lint

# Development commands
install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) application.py

train:
	$(PYTHON) -m src.train

test:
	$(PYTHON) -m pytest

smoke:
	$(SMOKE_CMD)

lint:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m black --check .

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

# Docker commands
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME)

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

# Development with Docker
docker-dev: docker-compose-up

# CI/CD helpers
ci: install lint test

# All-in-one local setup
setup: clean install train test

# Production deployment helpers
prod-build: clean docker-build

# Default target
all: setup

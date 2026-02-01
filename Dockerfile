FROM python:3.12-slim

WORKDIR /app

# Install curl for HEALTHCHECK and socat for optional port forwarding
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl socat \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5001

# Optionally pre-train at build time to avoid startup delays
ARG RUN_TRAINING=1
RUN if [ "$RUN_TRAINING" = "1" ]; then python -m src.train; fi
# Cache artifacts inside the image for fast startup with volumes
RUN if [ "$RUN_TRAINING" = "1" ] && [ -d /app/artifacts ]; then \
    mkdir -p /app/.artifacts_cache && cp -r /app/artifacts/* /app/.artifacts_cache/; \
  fi

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:5001/health || exit 1

RUN chmod +x /app/scripts/docker-entrypoint.sh
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["python", "application.py"]

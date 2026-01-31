FROM python:3.12-slim

WORKDIR /app

# Install curl for HEALTHCHECK + clean up apt cache
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Healthcheck hits your Flask /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:5000/health || exit 1

# Run the app
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "application.py"]

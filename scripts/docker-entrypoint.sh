#!/bin/sh
set -e

missing_files=""
for f in /app/artifacts/schema.json \
         /app/artifacts/preprocessor.pkl \
         /app/artifacts/encoder.pkl \
         /app/artifacts/model.pkl; do
  if [ ! -f "$f" ]; then
    missing_files="$missing_files $f"
  fi
done

if [ -n "$missing_files" ]; then
  if [ -d /app/.artifacts_cache ] && [ "$(ls -A /app/.artifacts_cache)" ]; then
    echo "Artifacts missing. Restoring from image cache..."
    mkdir -p /app/artifacts
    cp -r /app/.artifacts_cache/* /app/artifacts/
  elif [ "${AUTO_TRAIN:-1}" = "1" ]; then
    if [ "${AUTO_TRAIN_ASYNC:-0}" = "1" ]; then
      echo "Artifacts missing. AUTO_TRAIN_ASYNC=1 set, but blocking startup for readiness."
    fi
    echo "Artifacts missing. Training model..."
    python -m src.train
  else
    echo "Artifacts missing and AUTO_TRAIN=0. Run 'python -m src.train' or set AUTO_TRAIN=1."
    exit 1
  fi
fi

missing_files=""
for f in /app/artifacts/schema.json \
         /app/artifacts/preprocessor.pkl \
         /app/artifacts/encoder.pkl \
         /app/artifacts/model.pkl; do
  if [ ! -f "$f" ]; then
    missing_files="$missing_files $f"
  fi
done

if [ -n "$missing_files" ]; then
  echo "Artifacts still missing after startup checks:$missing_files"
  echo "Run 'python -m src.train' or rebuild with RUN_TRAINING=1."
  exit 1
fi

if [ "${ENABLE_PORT_5000:-1}" = "1" ] && [ "${PORT:-5001}" != "5000" ]; then
  echo "Forwarding port 5000 -> ${PORT:-5001}"
  socat TCP-LISTEN:5000,fork TCP:127.0.0.1:${PORT:-5001} &
fi

exec "$@"

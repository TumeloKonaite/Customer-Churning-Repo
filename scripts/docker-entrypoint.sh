#!/bin/sh
set -e

if [ ! -f /app/artifacts/schema.json ] || \
   [ ! -f /app/artifacts/preprocessor.pkl ] || \
   [ ! -f /app/artifacts/encoder.pkl ]; then
  if [ -d /app/.artifacts_cache ] && [ "$(ls -A /app/.artifacts_cache)" ]; then
    echo "Artifacts missing. Restoring from image cache..."
    mkdir -p /app/artifacts
    cp -r /app/.artifacts_cache/* /app/artifacts/
  elif [ "${AUTO_TRAIN:-1}" = "1" ]; then
    if [ "${AUTO_TRAIN_ASYNC:-1}" = "1" ]; then
      echo "Artifacts missing. Training model in background..."
      python -m src.train &
    else
      echo "Artifacts missing. Training model..."
      python -m src.train
    fi
  else
    echo "Artifacts missing and AUTO_TRAIN=0. Skipping training."
  fi
fi

if [ "${ENABLE_PORT_5000:-1}" = "1" ] && [ "${PORT:-5001}" != "5000" ]; then
  echo "Forwarding port 5000 -> ${PORT:-5001}"
  socat TCP-LISTEN:5000,fork TCP:127.0.0.1:${PORT:-5001} &
fi

exec "$@"

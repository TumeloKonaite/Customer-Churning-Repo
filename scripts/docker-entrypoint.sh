#!/bin/sh
set -e

if [ ! -f /app/artifacts/schema.json ] || \
   [ ! -f /app/artifacts/preprocessor.pkl ] || \
   [ ! -f /app/artifacts/encoder.pkl ]; then
  if [ "${AUTO_TRAIN:-1}" = "1" ]; then
    echo "Artifacts missing. Training model..."
    python -m src.train
  else
    echo "Artifacts missing and AUTO_TRAIN=0. Skipping training."
  fi
fi

exec "$@"

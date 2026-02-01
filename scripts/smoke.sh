#!/bin/sh
set -eu

BASE_URL="${BASE_URL:-http://localhost:5001}"
HEALTH_URL="${BASE_URL}/health"
PREDICT_URL="${BASE_URL}/api/predict"

echo "Smoke test against ${BASE_URL}"

health_body="$(mktemp)"
predict_body="$(mktemp)"
cleanup() {
  rm -f "$health_body" "$predict_body"
}
trap cleanup EXIT

health_status="$(curl -sS -o "$health_body" -w "%{http_code}" "$HEALTH_URL" || true)"
if [ "$health_status" -ne 200 ]; then
  echo "Health check failed (status ${health_status})."
  cat "$health_body"
  exit 1
fi
if ! grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"' "$health_body"; then
  echo "Health check did not return status=healthy."
  cat "$health_body"
  exit 1
fi

predict_status="$(curl -sS -o "$predict_body" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }' \
  "$PREDICT_URL" || true)"

if [ "$predict_status" -ne 200 ]; then
  echo "Predict request failed (status ${predict_status})."
  cat "$predict_body"
  exit 1
fi
if ! grep -q '"p_churn"' "$predict_body"; then
  echo "Predict response missing p_churn."
  cat "$predict_body"
  exit 1
fi

echo "Smoke test passed."

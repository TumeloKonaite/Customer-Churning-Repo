#!/bin/sh
set -eu

BASE_URL="${BASE_URL:-http://localhost:5001}"
HEALTH_URL="${BASE_URL}/health"
PREDICT_URL="${BASE_URL}/api/predict"
DOCS_URL="${BASE_URL}/docs"
OPENAPI_URL="${BASE_URL}/openapi.json"

echo "Smoke test against ${BASE_URL}"

health_body="$(mktemp)"
predict_body="$(mktemp)"
docs_body="$(mktemp)"
openapi_body="$(mktemp)"
cleanup() {
  rm -f "$health_body" "$predict_body" "$docs_body" "$openapi_body"
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
if ! grep -q '"model_loaded"[[:space:]]*:[[:space:]]*true' "$health_body"; then
  echo "Health check indicates model artifacts are not loaded."
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

docs_status="$(curl -sS -o "$docs_body" -w "%{http_code}" "$DOCS_URL" || true)"
if [ "$docs_status" -ne 200 ]; then
  echo "Swagger UI check failed (status ${docs_status})."
  cat "$docs_body"
  exit 1
fi

openapi_status="$(curl -sS -o "$openapi_body" -w "%{http_code}" "$OPENAPI_URL" || true)"
if [ "$openapi_status" -ne 200 ]; then
  echo "OpenAPI schema check failed (status ${openapi_status})."
  cat "$openapi_body"
  exit 1
fi
if ! grep -q '"openapi"' "$openapi_body"; then
  echo "OpenAPI response does not contain a schema document."
  cat "$openapi_body"
  exit 1
fi

echo "Smoke test passed."

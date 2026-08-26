from fastapi.testclient import TestClient

import application
from src.schemas.prediction import REQUIRED_FIELDS


client = TestClient(application.app)


def test_documentation_endpoints_are_available():
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["openapi"].startswith("3.")


def test_openapi_contains_every_public_api_operation():
    paths = client.get("/openapi.json").json()["paths"]
    expected = {
        "/health": {"get"},
        "/api/predict": {"post"},
        "/api/predict/batch": {"post"},
    }
    for path, methods in expected.items():
        assert methods <= set(paths[path])

    assert "/" not in paths
    assert "/predictdata" not in paths
    assert "/predictbatch" not in paths
    assert "/api/batch_predict" not in paths
    assert "/api/batch_predict_csv" not in paths


def test_json_request_schemas_document_model_fields_and_batch_options():
    document = client.get("/openapi.json").json()
    schemas = document["components"]["schemas"]
    single = schemas["SinglePredictionRequest"]
    assert set(REQUIRED_FIELDS) <= set(single["properties"])
    assert set(REQUIRED_FIELDS) <= set(single["required"])

    batch = schemas["BatchPredictionRequest"]
    assert {"records", "options"} <= set(batch["properties"])
    record = schemas["BatchPredictionRecord"]
    assert set(REQUIRED_FIELDS) <= set(record["properties"])
    assert {"customer_id", "row_id", "id"} <= set(record["properties"])
    options = schemas["BatchOptions"]
    assert set(options["properties"]["mode"]["enum"]) == {"fail_fast", "partial"}
    assert batch["properties"]["records"]["maxItems"] == 100


def test_success_and_error_response_schemas_are_documented():
    document = client.get("/openapi.json").json()
    paths = document["paths"]
    for path in ("/api/predict", "/api/predict/batch"):
        responses = paths[path]["post"]["responses"]
        assert "200" in responses
        assert "422" in responses
        assert responses["200"]["content"]["application/json"]["schema"]


def test_removed_legacy_html_prediction_routes_return_not_found():
    for path in ("/", "/predictdata", "/predictbatch"):
        assert client.get(path).status_code == 404

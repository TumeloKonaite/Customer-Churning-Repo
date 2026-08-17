"""OpenAPI compatibility customizations kept separate from app assembly."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.schemas.batch_prediction import (
    BatchOptions,
    BatchPredictionRecord,
    BatchPredictionRequest,
)
from src.schemas.errors import BatchContractError
from src.schemas.prediction import SinglePredictionRequest


def build_openapi_schema(app: FastAPI):
    """Document manually validated bodies without changing their runtime error contract."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in (
        SinglePredictionRequest,
        BatchPredictionRecord,
        BatchOptions,
        BatchPredictionRequest,
        BatchContractError,
    ):
        schemas[model.__name__] = model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )

    csv_body_schema = schema["paths"]["/api/batch_predict_csv"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]
    if "$ref" in csv_body_schema:
        csv_body_schema = schemas[csv_body_schema["$ref"].rsplit("/", 1)[-1]]
    csv_body_schema["properties"]["file"]["format"] = "binary"

    for path_item in schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.get("responses", {}).pop("422", None)

    app.openapi_schema = schema
    return app.openapi_schema

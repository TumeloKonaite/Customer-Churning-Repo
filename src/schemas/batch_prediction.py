"""Batch prediction request and response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.prediction import SINGLE_PREDICTION_EXAMPLE, SinglePredictionRequest


VALID_BATCH_MODES = {"fail_fast", "partial"}
MAX_BATCH_SIZE = 100
BATCH_PREDICTION_EXAMPLE = {
    "records": [{"customer_id": "CUST_001", **SINGLE_PREDICTION_EXAMPLE}],
    "options": {"mode": "partial"},
}


class BatchPredictionRecord(SinglePredictionRequest):
    """One batch record, with optional caller-provided identifier fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [BATCH_PREDICTION_EXAMPLE["records"][0]]},
    )
    customer_id: str | int | float | None = Field(
        default=None, description="Preferred optional record identifier"
    )
    row_id: str | int | float | None = Field(
        default=None, description="Optional fallback record identifier"
    )
    id: str | int | float | None = Field(
        default=None, description="Optional generic record identifier"
    )


class BatchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["fail_fast", "partial"] = Field(
        default="fail_fast",
        description="fail_fast stops at the first invalid record; partial scores all valid records",
    )


class BatchPredictionRequest(BaseModel):
    """The JSON batch contract, containing at most 100 records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [BATCH_PREDICTION_EXAMPLE]},
    )

    records: list[BatchPredictionRecord] = Field(max_length=MAX_BATCH_SIZE)
    options: BatchOptions = Field(default_factory=BatchOptions)


class BatchResultItem(BaseModel):
    index: int
    id: Any | None = Field(default=None, description="Passed through from customer_id, row_id, or id")
    predicted_label: int
    p_churn: float | None = Field(description="Churn probability, or null when unavailable")


class BatchValidationError(BaseModel):
    row_index: int
    id: Any | None = None
    message: str
    field: str | None = None


class BatchSummary(BaseModel):
    total_records: int
    valid_records: int
    invalid_records: int
    error_count: int
    mode: Literal["fail_fast", "partial"]


class BatchMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    model_version: str | None = None
    deployment_id: str | None = None
    model_version_id: str | None = None
    mlflow_run_id: str | None = None


class BatchResponse(BaseModel):
    status: Literal["success", "partial", "failed", "error"]
    results: list[BatchResultItem]
    errors: list[BatchValidationError] | None
    summary: BatchSummary
    metadata: BatchMetadata
    timestamp: str

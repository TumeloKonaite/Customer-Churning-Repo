"""Single-customer prediction schemas and shared feature contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.model_schema import CANONICAL_FEATURE_ORDER


SINGLE_PREDICTION_EXAMPLE = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}


class SinglePredictionRequest(BaseModel):
    """The ten customer features accepted by the churn model."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [SINGLE_PREDICTION_EXAMPLE]},
    )

    CreditScore: int = Field(ge=0, description="Customer credit score", examples=[619])
    Geography: str = Field(min_length=1, description="Customer country", examples=["France"])
    Gender: str = Field(min_length=1, description="Customer gender", examples=["Female"])
    Age: int = Field(ge=0, description="Customer age in years", examples=[42])
    Tenure: int = Field(ge=0, description="Years as a customer", examples=[2])
    Balance: float = Field(
        ge=0, allow_inf_nan=False, description="Account balance", examples=[0]
    )
    NumOfProducts: int = Field(
        ge=0, description="Number of bank products", examples=[1]
    )
    HasCrCard: int = Field(
        ge=0,
        le=1,
        description="Whether the customer has a credit card (0 or 1)", examples=[1]
    )
    IsActiveMember: int = Field(
        ge=0,
        le=1,
        description="Whether the customer is active (0 or 1)", examples=[1]
    )
    EstimatedSalary: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Estimated annual salary",
        examples=[101348.88],
    )


# Keep validation, CSV ingestion, and OpenAPI generation on one canonical field list.
REQUIRED_FIELDS = list(CANONICAL_FEATURE_ORDER)


class SinglePredictionResponse(BaseModel):
    status: Literal["success"]
    predicted_label: int
    p_churn: float | None = Field(description="Churn probability, or null when unavailable")
    model_name: str
    model_version: str
    timestamp: str

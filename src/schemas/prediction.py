"""Single-customer prediction schemas and shared feature contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(json_schema_extra={"examples": [SINGLE_PREDICTION_EXAMPLE]})

    CreditScore: float = Field(description="Customer credit score", examples=[619])
    Geography: str = Field(description="Customer country", examples=["France"])
    Gender: str = Field(description="Customer gender", examples=["Female"])
    Age: float = Field(description="Customer age in years", examples=[42])
    Tenure: float = Field(description="Years as a customer", examples=[2])
    Balance: float = Field(description="Account balance", examples=[0])
    NumOfProducts: float = Field(description="Number of bank products", examples=[1])
    HasCrCard: float = Field(
        description="Whether the customer has a credit card (0 or 1)", examples=[1]
    )
    IsActiveMember: float = Field(
        description="Whether the customer is active (0 or 1)", examples=[1]
    )
    EstimatedSalary: float = Field(description="Estimated annual salary", examples=[101348.88])


# Keep validation, CSV ingestion, and OpenAPI generation on one canonical field list.
REQUIRED_FIELDS = list(SinglePredictionRequest.model_fields)


class SinglePredictionResponse(BaseModel):
    status: Literal["success"]
    predicted_label: int
    p_churn: float | None = Field(description="Churn probability, or null when unavailable")
    model_name: str
    model_version: str
    timestamp: str

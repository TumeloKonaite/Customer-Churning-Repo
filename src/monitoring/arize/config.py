"""Worker-only Arize settings. This module is never imported by FastAPI."""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator

from src.config import Environment, _Settings


class ArizeExportSettings(_Settings):
    environment: Environment = Field(
        default=Environment.PRODUCTION,
        validation_alias=AliasChoices("ARIZE_ENVIRONMENT", "APP_ENV"),
    )
    export_enabled: bool = Field(default=False, alias="ARIZE_EXPORT_ENABLED")
    api_key: SecretStr | None = Field(default=None, alias="ARIZE_API_KEY")
    space_id: SecretStr | None = Field(default=None, alias="ARIZE_SPACE_ID")
    model_name: str = Field(default="churn_predictor", alias="ARIZE_MODEL_NAME")
    batch_size: int = Field(default=500, ge=1, le=5000, alias="ARIZE_EXPORT_BATCH_SIZE")
    max_attempts: int = Field(default=8, ge=1, le=32, alias="ARIZE_EXPORT_MAX_ATTEMPTS")
    claim_timeout_minutes: int = Field(default=15, ge=1, le=1440, alias="ARIZE_EXPORT_CLAIM_TIMEOUT_MINUTES")
    initial_backoff_seconds: int = Field(default=60, ge=1, le=86400, alias="ARIZE_EXPORT_INITIAL_BACKOFF_SECONDS")
    max_backoff_seconds: int = Field(default=3600, ge=1, le=604800, alias="ARIZE_EXPORT_MAX_BACKOFF_SECONDS")
    privacy_approval_id: str | None = Field(default=None, alias="ARIZE_PRIVACY_APPROVAL_ID")

    @field_validator("api_key", "space_id", "privacy_approval_id", mode="before")
    @classmethod
    def blank_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_export_mode(self) -> "ArizeExportSettings":
        if self.model_name != "churn_predictor":
            raise ValueError("ARIZE_MODEL_NAME must be churn_predictor")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("ARIZE_EXPORT_MAX_BACKOFF_SECONDS must not be below the initial backoff")
        if self.export_enabled and (not self.api_key or not self.space_id):
            raise ValueError("enabled Arize export requires API key and space ID")
        if self.export_enabled and not self.privacy_approval_id:
            raise ValueError("enabled Arize export requires a privacy approval ID")
        return self

    def require_enabled(self) -> None:
        if not self.export_enabled:
            raise RuntimeError("Arize export is disabled")

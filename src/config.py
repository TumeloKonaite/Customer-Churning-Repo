"""Typed, secret-safe configuration for training, deployment, and persistence."""

from __future__ import annotations

from enum import StrEnum
import re
from urllib.parse import parse_qs, urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )


class DagsHubSettings(_Settings):
    """DagsHub-backed MLflow tracking and production registration."""

    enabled: bool = Field(default=False, alias="ENABLE_DAGSHUB_TRACKING")
    experiment_name: str = Field(
        default="customer-churn-production", alias="MLFLOW_EXPERIMENT_NAME"
    )
    dagshub_repo_owner: str | None = Field(default=None, alias="DAGSHUB_REPO_OWNER")
    dagshub_repo_name: str | None = Field(default=None, alias="DAGSHUB_REPO_NAME")
    dagshub_token: SecretStr | None = Field(default=None, alias="DAGSHUB_TOKEN")
    register_model: bool = Field(default=False, alias="ENABLE_MODEL_REGISTRATION")
    registered_model_name: str = Field(
        default="churn_predictor", alias="MLFLOW_REGISTERED_MODEL_NAME"
    )

    @field_validator(
        "dagshub_repo_owner",
        "dagshub_repo_name",
        "dagshub_token",
        mode="before",
    )
    @classmethod
    def empty_string_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_registration_mode(self) -> "DagsHubSettings":
        if self.registered_model_name != "churn_predictor":
            raise ValueError("MLFLOW_REGISTERED_MODEL_NAME must be churn_predictor")
        if self.register_model and not self.enabled:
            raise ValueError(
                "ENABLE_MODEL_REGISTRATION requires ENABLE_DAGSHUB_TRACKING"
            )
        return self


class DatabaseSettings(_Settings):
    """Operational database settings. The URL stays redacted in representations."""

    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="APP_ENV")
    database_url: SecretStr | None = Field(default=None, alias="DATABASE_URL")
    connect_timeout_seconds: int = Field(
        default=10, ge=1, le=60, alias="DATABASE_CONNECT_TIMEOUT_SECONDS"
    )
    pool_size: int = Field(default=2, ge=1, le=10, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=1, ge=0, le=10, alias="DATABASE_MAX_OVERFLOW")

    @model_validator(mode="after")
    def validate_database(self) -> "DatabaseSettings":
        if self.environment is Environment.PRODUCTION and self.database_url is None:
            raise ValueError("DATABASE_URL is required in production")
        if self.database_url is None:
            return self
        raw = self.database_url.get_secret_value()
        parsed = urlparse(raw)
        if parsed.scheme not in {"postgresql+psycopg", "postgresql"}:
            raise ValueError("DATABASE_URL must use PostgreSQL with psycopg")
        if self.environment is Environment.PRODUCTION:
            if parsed.hostname in {None, "localhost", "127.0.0.1", "::1"}:
                raise ValueError("Production DATABASE_URL must not target a local database")
            if parse_qs(parsed.query).get("sslmode") != ["require"]:
                raise ValueError("Production DATABASE_URL must set sslmode=require")
        return self

    def sqlalchemy_url(self) -> str:
        if self.database_url is None:
            raise ValueError("DATABASE_URL is not configured")
        raw = self.database_url.get_secret_value()
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)


class DeploymentSettings(_Settings):
    """Identity assertions supplied to an exact-version inference deployment."""

    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="APP_ENV")
    model_name: str = Field(
        default="churn_predictor", alias="MLFLOW_REGISTERED_MODEL_NAME"
    )
    model_version: str = Field(alias="MLFLOW_MODEL_VERSION")
    expected_run_id: str = Field(alias="EXPECTED_MLFLOW_RUN_ID")
    expected_model_version_id: str = Field(alias="EXPECTED_MODEL_VERSION_ID")
    expected_pipeline_sha256: str = Field(alias="EXPECTED_PIPELINE_SHA256")
    expected_artifact_manifest_sha256: str = Field(
        alias="EXPECTED_ARTIFACT_MANIFEST_SHA256"
    )
    deployment_package_dir: str = Field(
        default="build/model", alias="DEPLOYMENT_PACKAGE_DIR"
    )

    @field_validator("model_version")
    @classmethod
    def exact_numeric_version(cls, value: str) -> str:
        if not str(value).isdigit() or int(value) < 1:
            raise ValueError("MLFLOW_MODEL_VERSION must be an exact positive integer")
        return str(value)

    @field_validator(
        "expected_pipeline_sha256", "expected_artifact_manifest_sha256"
    )
    @classmethod
    def sha256_digest(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("Expected checksum must be a SHA-256 hex digest")
        return normalized

    @model_validator(mode="after")
    def identity_matches_version(self) -> "DeploymentSettings":
        if not self.expected_model_version_id.endswith(
            f":{self.model_name}:{self.model_version}"
        ):
            raise ValueError(
                "EXPECTED_MODEL_VERSION_ID must match the configured model name and version"
            )
        return self


def redact_secret(value: str) -> str:
    """Return a safe marker instead of attempting partial credential disclosure."""
    return "<redacted>" if value else ""


def safe_error_message(error: BaseException) -> str:
    """Redact URL userinfo, bearer tokens, and common secret assignments."""
    message = str(error)
    message = re.sub(
        r"([a-z][a-z0-9+.-]*://)([^@/\s]+)@",
        r"\1<redacted>@",
        message,
        flags=re.IGNORECASE,
    )
    message = re.sub(
        r"(?i)(bearer\s+)[a-z0-9._~+/=-]+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(
        r"(?i)((?:password|token|secret)\s*[=:]\s*)[^,;\s]+",
        r"\1<redacted>",
        message,
    )
    return message

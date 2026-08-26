"""Typed, secret-safe configuration for training, deployment, and persistence."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
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


class MonitoringSettings(_Settings):
    """Artifact-bucket and exact-model settings used only by monitoring workers."""

    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="APP_ENV")
    model_version_id: str = Field(alias="EXPECTED_MODEL_VERSION_ID")
    artifact_bucket: str | None = Field(default=None, alias="MONITORING_ARTIFACT_BUCKET")
    artifact_endpoint_url: str | None = Field(
        default=None, alias="MONITORING_ARTIFACT_ENDPOINT_URL"
    )
    artifact_region: str | None = Field(default=None, alias="MONITORING_ARTIFACT_REGION")
    local_artifact_dir: Path | None = Field(
        default=None, alias="MONITORING_LOCAL_ARTIFACT_DIR"
    )

    @model_validator(mode="after")
    def one_artifact_backend(self) -> "MonitoringSettings":
        if bool(self.artifact_bucket) == bool(self.local_artifact_dir):
            raise ValueError(
                "configure exactly one of MONITORING_ARTIFACT_BUCKET or "
                "MONITORING_LOCAL_ARTIFACT_DIR"
            )
        if self.environment is Environment.PRODUCTION and not self.artifact_bucket:
            raise ValueError("production monitoring requires MONITORING_ARTIFACT_BUCKET")
        return self


class OutcomeIngestionSettings(_Settings):
    """Secrets and allow-lists for the protected outcome ingestion boundary."""

    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="APP_ENV")
    token_secret: SecretStr = Field(alias="CUSTOMER_TOKEN_HMAC_SECRET")
    token_key_id: str = Field(min_length=1, alias="CUSTOMER_TOKEN_KEY_ID")
    ingestion_api_key: SecretStr = Field(alias="OUTCOME_INGESTION_API_KEY")
    allowed_real_source_namespaces: str = Field(
        default="customer-master", alias="OUTCOME_ALLOWED_REAL_SOURCES"
    )

    @model_validator(mode="after")
    def strong_secrets(self) -> "OutcomeIngestionSettings":
        if len(self.token_secret.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("CUSTOMER_TOKEN_HMAC_SECRET must contain at least 32 bytes")
        if len(self.ingestion_api_key.get_secret_value()) < 24:
            raise ValueError("OUTCOME_INGESTION_API_KEY must contain at least 24 characters")
        if not self.allowed_sources:
            raise ValueError("OUTCOME_ALLOWED_REAL_SOURCES must not be empty")
        return self

    @property
    def allowed_sources(self) -> frozenset[str]:
        return frozenset(
            value.strip()
            for value in self.allowed_real_source_namespaces.split(",")
            if value.strip()
        )


class LabelMaterializationSettings(_Settings):
    """Versioned inputs required by label materialization workers."""

    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="APP_ENV")
    required_outcome_sources_csv: str = Field(
        default="customer-master", alias="REQUIRED_OUTCOME_SOURCES"
    )
    label_contract_version: str = Field(default="1.0.0", alias="LABEL_CONTRACT_VERSION")
    label_contract_approved: bool = Field(
        default=False, alias="LABEL_CONTRACT_APPROVED"
    )
    horizon_days: int = Field(default=90, ge=1, alias="PREDICTION_HORIZON_DAYS")
    grace_period_days: int = Field(default=7, ge=0, alias="LABEL_GRACE_PERIOD_DAYS")

    @property
    def required_sources(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.strip()
                for value in self.required_outcome_sources_csv.split(",")
                if value.strip()
            )
        )

    @model_validator(mode="after")
    def complete_label_contract(self) -> "LabelMaterializationSettings":
        if not self.required_sources:
            raise ValueError("REQUIRED_OUTCOME_SOURCES must not be empty")
        if (
            self.environment is Environment.PRODUCTION
            and not self.label_contract_approved
        ):
            raise ValueError("production label monitoring requires an approved contract")
        return self


class OutcomeMonitoringSettings(LabelMaterializationSettings):
    """Additional inputs required by performance monitoring workers."""

    model_version_id: str = Field(alias="EXPECTED_MODEL_VERSION_ID")
    deployment_ids_csv: str = Field(alias="MONITORING_DEPLOYMENT_IDS")
    policy_version: str = Field(default="1.0.0", alias="MONITORING_POLICY_VERSION")
    performance_cohort_days: int = Field(
        default=30, ge=1, alias="PERFORMANCE_COHORT_DAYS"
    )
    classification_threshold: float = Field(
        default=0.5, ge=0, le=1, alias="DEPLOYED_CLASSIFICATION_THRESHOLD"
    )
    minimum_privacy_size: int = Field(
        default=20, ge=2, alias="MONITORING_MINIMUM_PRIVACY_SIZE"
    )

    @property
    def deployment_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.strip()
                for value in self.deployment_ids_csv.split(",")
                if value.strip()
            )
        )

    @model_validator(mode="after")
    def complete_identity(self) -> "OutcomeMonitoringSettings":
        if not self.deployment_ids:
            raise ValueError("MONITORING_DEPLOYMENT_IDS must not be empty")
        if self.environment is Environment.PRODUCTION and self.minimum_privacy_size < 20:
            raise ValueError("production minimum privacy size must be at least 20")
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

"""Artifact reads used to load approved Arize reference baselines."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class ArtifactStore(Protocol):
    def read_uri(self, uri: str) -> bytes: ...


@dataclass(slots=True)
class S3ArtifactStore:
    bucket: str
    endpoint_url: str | None = None
    region_name: str | None = None

    def _client(self):
        import boto3

        return boto3.client(
            "s3", endpoint_url=self.endpoint_url, region_name=self.region_name
        )

    def read_uri(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError("reference_dataset_uri must be an s3://bucket/key URI")
        response = self._client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return response["Body"].read()


@dataclass(slots=True)
class LocalArtifactStore:
    """Filesystem implementation for local baseline loading and tests."""

    root: Path

    def read_uri(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError("local artifact store only accepts file URIs")
        return Path(parsed.path if parsed.scheme else uri).read_bytes()


@dataclass(slots=True)
class MLflowArtifactStore:
    """Read an immutable run artifact through the configured MLflow backend."""

    def read_uri(self, uri: str) -> bytes:
        match = re.fullmatch(r"runs:/([0-9a-fA-F]{32})/(.+)", uri)
        if not match or ".." in Path(match.group(2)).parts:
            raise ValueError(
                "MLflow artifact store requires a runs:/<run-id>/<artifact-path> URI"
            )

        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=uri,
                    dst_path=directory,
                )
            )
            if not downloaded.is_file():
                raise ValueError("MLflow artifact URI did not resolve to a file")
            return downloaded.read_bytes()

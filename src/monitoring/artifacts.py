"""Immutable artifact-bucket reads and monitoring report publication."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from src.monitoring.models import canonical_json_bytes, sha256_bytes


class ArtifactConflictError(RuntimeError):
    pass


class ArtifactStore(Protocol):
    def read_uri(self, uri: str) -> bytes: ...
    def put_immutable(self, key: str, body: bytes, content_type: str) -> str: ...


def artifact_prefix(model_version_id: str, baseline_version_id: str, run_id: str) -> str:
    segments: list[str] = []
    for value in (model_version_id, baseline_version_id, run_id):
        if not value or value in {".", ".."}:
            raise ValueError("artifact path identities must be non-empty")
        segments.append(quote(value, safe="-._~:"))
    return f"monitoring/{segments[0]}/{segments[1]}/drift/{segments[2]}"


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

    def put_immutable(self, key: str, body: bytes, content_type: str) -> str:
        normalized = PurePosixPath(key).as_posix()
        if normalized != key or key.startswith("/") or ".." in PurePosixPath(key).parts:
            raise ValueError("artifact key is not normalized")
        checksum = sha256_bytes(body)
        client = self._client()
        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata={"sha256": checksum},
                IfNoneMatch="*",
            )
        except Exception as exc:
            try:
                existing = client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except Exception:
                raise exc
            if sha256_bytes(existing) != checksum:
                raise ArtifactConflictError(f"immutable artifact conflict at {key}") from exc
        return f"s3://{self.bucket}/{key}"


@dataclass(slots=True)
class LocalArtifactStore:
    """Filesystem implementation for local debugging and unit tests."""

    root: Path

    def read_uri(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError("local artifact store only accepts file URIs")
        return Path(parsed.path if parsed.scheme else uri).read_bytes()

    def put_immutable(self, key: str, body: bytes, content_type: str) -> str:
        del content_type
        path = self.root.joinpath(*PurePosixPath(key).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != body:
                raise ArtifactConflictError(f"immutable artifact conflict at {key}")
        else:
            path.write_bytes(body)
        return path.resolve().as_uri()


def publish_report_bundle(
    store: ArtifactStore,
    *,
    prefix: str,
    html: bytes,
    report: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    payloads = {
        "report.html": (html, "text/html; charset=utf-8"),
        "report.json": (canonical_json_bytes(report), "application/json"),
        "summary.json": (canonical_json_bytes(summary), "application/json"),
    }
    checksums = {name: sha256_bytes(body) for name, (body, _) in payloads.items()}
    payloads["checksums.json"] = (
        canonical_json_bytes({"algorithm": "sha256", "artifacts": checksums}),
        "application/json",
    )
    uris = {
        name: store.put_immutable(f"{prefix}/{name}", body, content_type)
        for name, (body, content_type) in payloads.items()
    }
    return {"uris": uris, "checksums": checksums}


def publish_summary_bundle(
    store: ArtifactStore, *, prefix: str, summary: dict[str, Any]
) -> dict[str, Any]:
    body = canonical_json_bytes(summary)
    checksums = {"summary.json": sha256_bytes(body)}
    checksum_body = canonical_json_bytes(
        {"algorithm": "sha256", "artifacts": checksums}
    )
    uris = {
        "summary.json": store.put_immutable(
            f"{prefix}/summary.json", body, "application/json"
        ),
        "checksums.json": store.put_immutable(
            f"{prefix}/checksums.json", checksum_body, "application/json"
        ),
    }
    return {"uris": uris, "checksums": checksums}

"""Shared API error schemas."""

from typing import Literal

from pydantic import BaseModel


BATCH_CONTRACT_VERSION = "v1"


class StandardAPIError(BaseModel):
    status: Literal["error"]
    message: str
    errors: list[str] | None = None


class BatchContractError(BaseModel):
    status: Literal["error"]
    message: str
    contract_version: Literal["v1"]

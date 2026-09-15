"""Authoritative outcome boundary and persistence models.

Definitions remain in ``service`` for compatibility with the original cohesive
implementation; this module is the stable model import boundary for consumers.
"""

from src.monitoring.outcomes.service import (
    CanonicalOutcome,
    OutcomeIngestionRequest,
    OutcomeOperation,
    OutcomeReference,
)

__all__ = [
    "CanonicalOutcome",
    "OutcomeIngestionRequest",
    "OutcomeOperation",
    "OutcomeReference",
]


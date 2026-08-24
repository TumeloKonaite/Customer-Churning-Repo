"""Pure deterministic extraction-window and row-selection rules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from src.monitoring.models import (
    ExtractionWatermark,
    MonitoringPolicy,
    PredictionRecord,
    SelectedWindow,
    require_utc,
)


CountRows = Callable[[datetime, datetime, ExtractionWatermark], int]


def extraction_cutoff(scheduled_for: datetime, policy: MonitoringPolicy) -> datetime:
    """Return a cadence-stable cutoff; callers persist its database cursor separately."""
    return require_utc(scheduled_for, "scheduled_for") - policy.data_latency_allowance


def select_window(
    *,
    end: datetime,
    watermark: ExtractionWatermark,
    policy: MonitoringPolicy,
    count_rows: CountRows,
) -> SelectedWindow:
    """Expand geometrically until volume is sufficient or the hard boundary is hit."""
    end = require_utc(end, "end")
    lookback = policy.initial_lookback
    maximum_start = end - policy.maximum_lookback
    hard_start = maximum_start
    if policy.fixed_historical_boundary is not None:
        hard_start = max(hard_start, policy.fixed_historical_boundary)

    while True:
        start = max(end - lookback, hard_start)
        observed = count_rows(start, end, watermark)
        reached_boundary = start == hard_start
        if observed >= policy.minimum_current_rows or reached_boundary:
            return SelectedWindow(
                start=start,
                end=end,
                observed_rows=observed,
                selected_rows=min(observed, policy.maximum_current_rows),
                lookback_hours=int((end - start).total_seconds() // 3600),
                reached_boundary=reached_boundary,
                deterministic_limit_applied=observed > policy.maximum_current_rows,
            )
        lookback = min(lookback * 2, policy.maximum_lookback)


def eligible_record(
    record: PredictionRecord,
    *,
    environment: str,
    model_version_id: str,
    window: SelectedWindow,
    watermark: ExtractionWatermark,
) -> bool:
    return (
        record.environment == environment
        and record.model_version_id == model_version_id
        and window.start <= record.prediction_timestamp <= window.end
        and record.persisted_at <= watermark.extraction_cutoff
        and record.event_id <= watermark.maximum_persisted_event_id
    )


def deterministically_limit(
    records: Iterable[PredictionRecord], maximum_rows: int
) -> tuple[PredictionRecord, ...]:
    """Keep the newest bounded cohort, with event_id as the stable tie breaker."""
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive")
    newest = sorted(
        records,
        key=lambda row: (row.prediction_timestamp, row.event_id),
        reverse=True,
    )[:maximum_rows]
    return tuple(sorted(newest, key=lambda row: (row.prediction_timestamp, row.event_id)))


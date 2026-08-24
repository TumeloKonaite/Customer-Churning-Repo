"""Add outcome ingestion, label revisions, and performance reporting.

Revision ID: 20260824_0003
Revises: 20260822_0002
Create Date: 2026-08-24 00:00:00+00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260824_0003"
down_revision = "20260822_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Legacy prediction rows remain explicitly ineligible until their identity and
    # horizon can be sourced authoritatively; they are never silently backfilled.
    op.add_column("prediction_events", sa.Column("customer_token", sa.Text(), nullable=True))
    op.add_column("prediction_events", sa.Column("token_key_id", sa.Text(), nullable=True))
    op.add_column(
        "prediction_events", sa.Column("horizon_end", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("prediction_events", sa.Column("label_contract_version", sa.Text(), nullable=True))
    op.add_column("prediction_events", sa.Column("deployment_id", sa.Text(), nullable=True))
    op.add_column("prediction_events", sa.Column("policy_version", sa.Text(), nullable=True))
    op.add_column(
        "prediction_events",
        sa.Column("monitoring_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "prediction_events",
        sa.Column(
            "segments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_check_constraint(
        "ck_prediction_horizon_after_prediction",
        "prediction_events",
        "horizon_end IS NULL OR horizon_end > prediction_timestamp",
    )
    op.create_check_constraint(
        "ck_prediction_monitoring_identity_complete",
        "prediction_events",
        "NOT monitoring_eligible OR (customer_token IS NOT NULL AND "
        "token_key_id IS NOT NULL AND horizon_end IS NOT NULL AND "
        "label_contract_version IS NOT NULL AND deployment_id IS NOT NULL AND "
        "policy_version IS NOT NULL)",
    )
    op.create_index(
        "ix_prediction_events_customer_horizon",
        "prediction_events",
        ["environment", "customer_token", "prediction_timestamp", "horizon_end"],
    )

    op.create_table(
        "outcome_events",
        sa.Column("outcome_ingest_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("outcome_event_id", sa.String(length=44), nullable=False, unique=True),
        sa.Column("source_event_id", sa.Text(), nullable=False),
        sa.Column("source_namespace", sa.Text(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("customer_token", sa.Text(), nullable=False),
        sa.Column("token_key_id", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column(
            "referenced_outcome_event_id",
            sa.String(length=44),
            sa.ForeignKey("outcome_events.outcome_event_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        sa.Column("simulation_generator", sa.Text(), nullable=True),
        sa.Column("simulation_scenario_version", sa.Text(), nullable=True),
        sa.Column("label_contract_version", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "persisted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.UniqueConstraint(
            "source_namespace", "source_event_id", name="uq_outcome_source_identity"
        ),
        sa.CheckConstraint(
            "event_timestamp <= received_timestamp", name="ck_outcome_timestamp_order"
        ),
        sa.CheckConstraint(
            "operation IN ('create','correction','retraction','supersession')",
            name="ck_outcome_operation",
        ),
        sa.CheckConstraint(
            "(operation = 'create' AND referenced_outcome_event_id IS NULL) OR "
            "(operation <> 'create' AND referenced_outcome_event_id IS NOT NULL)",
            name="ck_outcome_reference_operation",
        ),
        sa.CheckConstraint(
            "(is_simulated AND simulation_generator IS NOT NULL AND "
            "simulation_scenario_version IS NOT NULL) OR "
            "(NOT is_simulated AND simulation_generator IS NULL AND "
            "simulation_scenario_version IS NULL)",
            name="ck_outcome_simulation_metadata",
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[a-f0-9]{64}$'", name="ck_outcome_content_sha256"
        ),
    )
    op.create_index(
        "ix_outcomes_customer_attribution",
        "outcome_events",
        ["environment", "is_simulated", "customer_token", "event_timestamp"],
    )
    op.create_index(
        "ix_outcomes_snapshot",
        "outcome_events",
        ["persisted_at", "outcome_ingest_id"],
    )
    op.create_index(
        "uq_outcome_single_superseding_child",
        "outcome_events",
        ["referenced_outcome_event_id"],
        unique=True,
        postgresql_where=sa.text("referenced_outcome_event_id IS NOT NULL"),
    )

    op.create_table(
        "outcome_quarantine",
        sa.Column("quarantine_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("source_namespace", sa.Text(), nullable=True),
        sa.Column("source_event_id", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "quarantined_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("clock_timestamp()")
        ),
    )

    op.create_table(
        "outcome_source_watermarks",
        sa.Column("watermark_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("source_namespace", sa.Text(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        sa.Column("complete_through", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_namespace", "environment", "is_simulated", "complete_through",
            name="uq_outcome_source_watermark_declaration",
        ),
    )
    op.create_index(
        "ix_outcome_watermarks_snapshot",
        "outcome_source_watermarks",
        ["environment", "is_simulated", "source_namespace", "observed_at"],
    )

    op.create_table(
        "label_materialization_runs",
        sa.Column("materialization_run_id", sa.String(length=71), primary_key=True),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        sa.Column("simulation_generator", sa.Text(), nullable=True),
        sa.Column("simulation_scenario_version", sa.Text(), nullable=True),
        sa.Column("label_contract_version", sa.Text(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("grace_period_days", sa.Integer(), nullable=False),
        sa.Column("outcome_watermark", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("horizon_days > 0", name="ck_label_run_horizon"),
        sa.CheckConstraint("grace_period_days >= 0", name="ck_label_run_grace"),
        sa.CheckConstraint(
            "(is_simulated AND simulation_generator IS NOT NULL AND "
            "simulation_scenario_version IS NOT NULL) OR "
            "(NOT is_simulated AND simulation_generator IS NULL AND "
            "simulation_scenario_version IS NULL)",
            name="ck_label_run_simulation_metadata",
        ),
    )

    op.create_table(
        "prediction_label_revisions",
        sa.Column("label_revision_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "prediction_id", sa.Text(),
            sa.ForeignKey("prediction_events.prediction_id", ondelete="RESTRICT"), nullable=False
        ),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("label_value", sa.SmallInteger(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "qualifying_outcome_event_id", sa.String(length=44),
            sa.ForeignKey("outcome_events.outcome_event_id", ondelete="RESTRICT"), nullable=True
        ),
        sa.Column("label_contract_version", sa.Text(), nullable=False),
        sa.Column(
            "materialization_run_id", sa.String(length=71),
            sa.ForeignKey("label_materialization_runs.materialization_run_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("attribution_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "supersedes_label_revision_id", sa.BigInteger(),
            sa.ForeignKey("prediction_label_revisions.label_revision_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("revision_reason", sa.Text(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        sa.Column("simulation_generator", sa.Text(), nullable=True),
        sa.Column("simulation_scenario_version", sa.Text(), nullable=True),
        sa.Column("simulation_scope", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "prediction_id", "revision_number", "simulation_scope",
            name="uq_label_prediction_revision_mode",
        ),
        sa.UniqueConstraint(
            "prediction_id", "materialization_run_id", "simulation_scope",
            name="uq_label_prediction_run_mode",
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_label_revision_positive"),
        sa.CheckConstraint("label_value IS NULL OR label_value IN (0,1)", name="ck_label_binary"),
        sa.CheckConstraint(
            "status IN ('positive','negative','pending')", name="ck_label_status"
        ),
        sa.CheckConstraint(
            "(status = 'positive' AND label_value = 1 AND qualifying_outcome_event_id IS NOT NULL) OR "
            "(status = 'negative' AND label_value = 0 AND qualifying_outcome_event_id IS NULL) OR "
            "(status = 'pending' AND label_value IS NULL AND "
            "qualifying_outcome_event_id IS NULL)",
            name="ck_label_state_content",
        ),
        sa.CheckConstraint(
            "(is_simulated AND simulation_generator IS NOT NULL AND "
            "simulation_scenario_version IS NOT NULL AND simulation_scope <> 'real') OR "
            "(NOT is_simulated AND simulation_generator IS NULL AND "
            "simulation_scenario_version IS NULL AND simulation_scope = 'real')",
            name="ck_label_simulation_scope",
        ),
    )
    op.create_index(
        "ix_label_revision_snapshot",
        "prediction_label_revisions",
        ["simulation_scope", "prediction_id", sa.text("revision_number DESC")],
    )

    op.create_table(
        "performance_monitoring_runs",
        sa.Column("monitoring_run_id", sa.String(length=76), primary_key=True),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("deployment_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("label_contract_version", sa.Text(), nullable=False),
        sa.Column("cohort_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cohort_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cohort_selection_rule", sa.Text(), nullable=False),
        sa.Column("outcome_watermark", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("label_revision_watermark", sa.BigInteger(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        sa.Column("simulation_generator", sa.Text(), nullable=True),
        sa.Column("simulation_scenario_version", sa.Text(), nullable=True),
        sa.Column("artifact_prefix", sa.Text(), nullable=False, unique=True),
        sa.Column("run_configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("artifact_uris", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("artifact_checksums", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("suppression_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("cohort_end > cohort_start", name="ck_performance_cohort_interval"),
        sa.CheckConstraint(
            "(is_simulated AND simulation_generator IS NOT NULL AND "
            "simulation_scenario_version IS NOT NULL) OR "
            "(NOT is_simulated AND simulation_generator IS NULL AND "
            "simulation_scenario_version IS NULL)",
            name="ck_performance_simulation_metadata",
        ),
    )

    # Customer-level facts and label history are append-only. Report/run rows may
    # transition status but artifacts are protected by immutable object-store writes.
    op.execute(
        """
        CREATE FUNCTION reject_monitoring_fact_update() RETURNS trigger AS $$
        BEGIN
          RAISE EXCEPTION 'monitoring facts are append-only';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    for table in ("outcome_events", "outcome_source_watermarks", "prediction_label_revisions"):
        op.execute(
            f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION reject_monitoring_fact_update()"
        )


def downgrade() -> None:
    for table in ("prediction_label_revisions", "outcome_source_watermarks", "outcome_events"):
        op.execute(f"DROP TRIGGER {table}_append_only ON {table}")
    op.execute("DROP FUNCTION reject_monitoring_fact_update()")
    op.drop_table("performance_monitoring_runs")
    op.drop_table("prediction_label_revisions")
    op.drop_table("label_materialization_runs")
    op.drop_index("ix_outcome_watermarks_snapshot", table_name="outcome_source_watermarks")
    op.drop_table("outcome_source_watermarks")
    op.drop_table("outcome_quarantine")
    op.drop_index("uq_outcome_single_superseding_child", table_name="outcome_events")
    op.drop_index("ix_outcomes_snapshot", table_name="outcome_events")
    op.drop_index("ix_outcomes_customer_attribution", table_name="outcome_events")
    op.drop_table("outcome_events")
    op.drop_index("ix_prediction_events_customer_horizon", table_name="prediction_events")
    op.drop_constraint("ck_prediction_horizon_after_prediction", "prediction_events", type_="check")
    op.drop_constraint(
        "ck_prediction_monitoring_identity_complete", "prediction_events", type_="check"
    )
    for column in (
        "segments", "monitoring_eligible", "policy_version", "deployment_id",
        "label_contract_version", "horizon_end", "token_key_id", "customer_token",
    ):
        op.drop_column("prediction_events", column)

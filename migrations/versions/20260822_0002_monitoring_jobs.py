"""Add immutable monitoring policies, baselines, prediction events, and runs.

Revision ID: 20260822_0002
Revises: 20260820_0001
Create Date: 2026-08-22 00:00:00+00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260822_0002"
down_revision = "20260820_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "monitoring_policies",
        sa.Column("policy_version", sa.Text(), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("configuration_sha256", sa.String(length=64), nullable=False, unique=True),
        sa.Column("included_environments", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("included_model_versions", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("policy_version <> ''", name="ck_monitoring_policy_version"),
        sa.CheckConstraint(
            "configuration_sha256 ~ '^[a-f0-9]{64}$'",
            name="ck_monitoring_policy_sha256",
        ),
    )
    op.create_index(
        "ix_monitoring_policies_enabled_created",
        "monitoring_policies",
        ["enabled", "created_at"],
    )

    op.create_table(
        "monitoring_baselines",
        sa.Column("baseline_version_id", sa.Text(), primary_key=True),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("reference_dataset_uri", sa.Text(), nullable=False),
        sa.Column("reference_sha256", sa.String(length=64), nullable=False),
        sa.Column("feature_schema_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("approval_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("baseline_version_id <> ''", name="ck_monitoring_baseline_id"),
        sa.CheckConstraint("model_version_id <> ''", name="ck_monitoring_baseline_model"),
        sa.CheckConstraint(
            "reference_sha256 ~ '^[a-f0-9]{64}$'",
            name="ck_monitoring_baseline_sha256",
        ),
        sa.CheckConstraint(
            "retired_at IS NULL OR retired_at > active_from",
            name="ck_monitoring_baseline_interval",
        ),
        sa.CheckConstraint(
            "active_from >= created_at",
            name="ck_monitoring_baseline_not_retroactive",
        ),
    )
    op.create_index(
        "ix_monitoring_baselines_model_interval",
        "monitoring_baselines",
        ["model_version_id", "active_from", "retired_at"],
    )

    op.create_table(
        "prediction_events",
        sa.Column("event_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("prediction_id", sa.Text(), nullable=False, unique=True),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "persisted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("feature_schema_version", sa.Text(), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prediction_probability", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "prediction_probability >= 0 AND prediction_probability <= 1",
            name="ck_prediction_probability_range",
        ),
    )
    op.create_index(
        "ix_prediction_events_monitoring_extract",
        "prediction_events",
        [
            "environment",
            "model_version_id",
            sa.text("prediction_timestamp DESC"),
            sa.text("event_id DESC"),
        ],
    )
    op.create_index(
        "ix_prediction_events_persisted_cursor",
        "prediction_events",
        ["persisted_at", "event_id"],
    )
    op.execute(
        """
        CREATE FUNCTION stamp_prediction_event_arrival() RETURNS trigger AS $$
        BEGIN
          NEW.persisted_at := clock_timestamp();
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER prediction_event_arrival BEFORE INSERT ON prediction_events "
        "FOR EACH ROW EXECUTE FUNCTION stamp_prediction_event_arrival()"
    )
    op.execute(
        """
        CREATE FUNCTION reject_prediction_event_update() RETURNS trigger AS $$
        BEGIN
          RAISE EXCEPTION 'prediction events are append-only';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER prediction_event_append_only BEFORE UPDATE ON prediction_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_prediction_event_update()"
    )

    op.create_table(
        "monitoring_runs",
        sa.Column("monitoring_run_id", sa.String(length=73), primary_key=True),
        sa.Column("logical_job_key", sa.String(length=64), nullable=False, unique=True),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column(
            "baseline_version_id",
            sa.Text(),
            sa.ForeignKey("monitoring_baselines.baseline_version_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "policy_version",
            sa.Text(),
            sa.ForeignKey("monitoring_policies.policy_version", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("data_quality_status", sa.Text(), nullable=True),
        sa.Column("drift_status", sa.Text(), nullable=True),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("extraction_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("extraction_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extraction_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("maximum_persisted_event_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "maximum_eligible_prediction_timestamp",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_row_count", sa.Integer(), nullable=False),
        sa.Column("selected_row_count", sa.Integer(), nullable=False),
        sa.Column("selection_criteria", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("run_configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("evidently_version", sa.Text(), nullable=True),
        sa.Column("artifact_prefix", sa.Text(), nullable=False, unique=True),
        sa.Column("artifact_uris", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("artifact_checksums", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_kind", sa.Text(), nullable=True),
        sa.Column("error_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'insufficient_data', 'failed')",
            name="ck_monitoring_run_status",
        ),
        sa.CheckConstraint(
            "observed_row_count >= 0 AND selected_row_count >= 0 "
            "AND selected_row_count <= observed_row_count",
            name="ck_monitoring_run_counts",
        ),
        sa.CheckConstraint("window_start <= window_end", name="ck_monitoring_run_window"),
    )
    op.create_index(
        "ix_monitoring_runs_lookup",
        "monitoring_runs",
        ["environment", "model_version_id", "scheduled_for"],
    )

    # Result-affecting policy content and baseline dataset identity are append-only.
    op.execute(
        """
        CREATE FUNCTION reject_monitoring_identity_mutation() RETURNS trigger AS $$
        BEGIN
          IF TG_TABLE_NAME = 'monitoring_policies' AND
             (NEW.policy_version, NEW.configuration, NEW.configuration_sha256,
              NEW.included_environments, NEW.included_model_versions)
             IS DISTINCT FROM
             (OLD.policy_version, OLD.configuration, OLD.configuration_sha256,
              OLD.included_environments, OLD.included_model_versions) THEN
            RAISE EXCEPTION 'monitoring policy content is immutable; create a new version';
          END IF;
          IF TG_TABLE_NAME = 'monitoring_baselines' AND
             (NEW.baseline_version_id, NEW.model_version_id, NEW.reference_dataset_uri,
              NEW.reference_sha256, NEW.feature_schema_version, NEW.created_at,
              NEW.active_from, NEW.purpose, NEW.approval_metadata)
             IS DISTINCT FROM
             (OLD.baseline_version_id, OLD.model_version_id, OLD.reference_dataset_uri,
              OLD.reference_sha256, OLD.feature_schema_version, OLD.created_at,
              OLD.active_from, OLD.purpose, OLD.approval_metadata) THEN
            RAISE EXCEPTION 'monitoring baseline identity is immutable; create a new version';
          END IF;
          IF TG_TABLE_NAME = 'monitoring_baselines' AND OLD.retired_at IS NOT NULL
             AND NEW.retired_at IS DISTINCT FROM OLD.retired_at THEN
            RAISE EXCEPTION 'a baseline retirement timestamp cannot be changed';
          END IF;
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER monitoring_policy_immutable BEFORE UPDATE ON monitoring_policies "
        "FOR EACH ROW EXECUTE FUNCTION reject_monitoring_identity_mutation()"
    )
    op.execute(
        "CREATE TRIGGER monitoring_baseline_immutable BEFORE UPDATE ON monitoring_baselines "
        "FOR EACH ROW EXECUTE FUNCTION reject_monitoring_identity_mutation()"
    )
    op.execute(
        """
        CREATE FUNCTION reject_monitoring_version_delete() RETURNS trigger AS $$
        BEGIN
          RAISE EXCEPTION 'monitoring policy and baseline versions are append-only';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER monitoring_policy_no_delete BEFORE DELETE ON monitoring_policies "
        "FOR EACH ROW EXECUTE FUNCTION reject_monitoring_version_delete()"
    )
    op.execute(
        "CREATE TRIGGER monitoring_baseline_no_delete BEFORE DELETE ON monitoring_baselines "
        "FOR EACH ROW EXECUTE FUNCTION reject_monitoring_version_delete()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER monitoring_baseline_no_delete ON monitoring_baselines")
    op.execute("DROP TRIGGER monitoring_policy_no_delete ON monitoring_policies")
    op.execute("DROP FUNCTION reject_monitoring_version_delete()")
    op.execute("DROP TRIGGER monitoring_baseline_immutable ON monitoring_baselines")
    op.execute("DROP TRIGGER monitoring_policy_immutable ON monitoring_policies")
    op.execute("DROP FUNCTION reject_monitoring_identity_mutation()")
    op.drop_table("monitoring_runs")
    op.execute("DROP TRIGGER prediction_event_append_only ON prediction_events")
    op.execute("DROP FUNCTION reject_prediction_event_update()")
    op.execute("DROP TRIGGER prediction_event_arrival ON prediction_events")
    op.execute("DROP FUNCTION stamp_prediction_event_arrival()")
    op.drop_table("prediction_events")
    op.drop_table("monitoring_baselines")
    op.drop_table("monitoring_policies")

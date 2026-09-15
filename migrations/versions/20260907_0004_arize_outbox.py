"""Add the privacy-gated Arize transactional outbox.

Revision ID: 20260907_0004
Revises: 20260824_0003
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260907_0004"
down_revision = "20260824_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A normal UPDATE would correctly trip prediction_event_append_only. A
    # temporary constant default lets PostgreSQL materialize the legacy value as
    # part of the schema change without weakening that immutability trigger.
    op.add_column(
        "prediction_events",
        sa.Column(
            "request_source", sa.Text(), nullable=False,
            server_default=sa.text("'single'"),
        ),
    )
    op.add_column("prediction_events", sa.Column("batch_id", sa.Text(), nullable=True))
    op.alter_column("prediction_events", "request_source", server_default=None)
    op.create_check_constraint(
        "ck_prediction_request_source", "prediction_events",
        "request_source IN ('single','batch')",
    )
    op.create_table(
        "arize_privacy_approvals",
        sa.Column("approval_id", sa.Text(), primary_key=True),
        sa.Column("destination", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("approved_features", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("approved_tags", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint("destination = 'arize'", name="ck_arize_approval_destination"),
        sa.CheckConstraint("expires_at IS NULL OR expires_at > approved_at", name="ck_arize_approval_expiry"),
    )
    op.create_table(
        "arize_export_events",
        sa.Column("export_event_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("prediction_id", sa.Text(), sa.ForeignKey("prediction_events.prediction_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("label_revision_id", sa.BigInteger(), sa.ForeignKey("prediction_label_revisions.label_revision_id", ondelete="RESTRICT"), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claim_token", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("last_error_code", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.CheckConstraint("event_type IN ('prediction','actual')", name="ck_arize_event_type"),
        sa.CheckConstraint("status IN ('pending','processing','sent','dead_letter')", name="ck_arize_event_status"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_arize_attempt_count"),
        sa.CheckConstraint("(event_type = 'prediction' AND label_revision_id IS NULL) OR (event_type = 'actual' AND label_revision_id IS NOT NULL)", name="ck_arize_event_revision"),
    )
    op.create_index("uq_arize_prediction_export", "arize_export_events", ["prediction_id"], unique=True, postgresql_where=sa.text("event_type = 'prediction'"))
    op.create_index("uq_arize_actual_export", "arize_export_events", ["label_revision_id"], unique=True, postgresql_where=sa.text("event_type = 'actual'"))
    op.create_index("ix_arize_export_due", "arize_export_events", ["status", "next_attempt_at", "created_at", "export_event_id"])
    op.create_index("ix_arize_export_stale", "arize_export_events", ["status", "claimed_at"])
    op.create_table(
        "arize_baseline_exports",
        sa.Column("baseline_version_id", sa.Text(), sa.ForeignKey(
            "monitoring_baselines.baseline_version_id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("model_version_id", sa.Text(), nullable=False),
        sa.Column("reference_sha256", sa.String(length=64), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("baseline_version_id", "model_version_id",
                            name="uq_arize_baseline_model"),
    )


def downgrade() -> None:
    op.drop_table("arize_baseline_exports")
    op.drop_table("arize_export_events")
    op.drop_table("arize_privacy_approvals")
    op.drop_constraint("ck_prediction_request_source", "prediction_events", type_="check")
    op.drop_column("prediction_events", "batch_id")
    op.drop_column("prediction_events", "request_source")

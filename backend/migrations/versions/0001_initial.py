"""Initial schema: builds and build_events.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "builds",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column("owner", sqlmodel.sql.sqltypes.AutoString(length=100), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(length=20), nullable=False),
        sa.Column("request", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(length=100), nullable=True),
        sa.Column("agent_type", sqlmodel.sql.sqltypes.AutoString(length=20), nullable=True),
        sa.Column("failed_stage", sqlmodel.sql.sqltypes.AutoString(length=30), nullable=True),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("deployed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("result_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("builds") as batch:
        batch.create_index("ix_builds_owner", ["owner"])
        batch.create_index("ix_builds_status", ["status"])
        batch.create_index("ix_builds_deployed", ["deployed"])
        batch.create_index("ix_builds_created_at", ["created_at"])

    op.create_table(
        "build_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("build_id", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(length=10), nullable=False),
        sa.Column("stage", sqlmodel.sql.sqltypes.AutoString(length=30), nullable=True),
        sa.Column("event", sqlmodel.sql.sqltypes.AutoString(length=30), nullable=False),
        sa.Column("message", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["build_id"], ["builds.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("build_events") as batch:
        batch.create_index("ix_build_events_build_id", ["build_id"])


def downgrade() -> None:
    with op.batch_alter_table("build_events") as batch:
        batch.drop_index("ix_build_events_build_id")
    op.drop_table("build_events")
    with op.batch_alter_table("builds") as batch:
        batch.drop_index("ix_builds_created_at")
        batch.drop_index("ix_builds_deployed")
        batch.drop_index("ix_builds_status")
        batch.drop_index("ix_builds_owner")
    op.drop_table("builds")

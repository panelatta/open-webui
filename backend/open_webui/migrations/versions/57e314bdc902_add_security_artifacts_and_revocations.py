"""Persist opaque artifact ownership and individual session revocation."""

import sqlalchemy as sa
from alembic import op

revision = "57e314bdc902"
down_revision = "b8f2c6d91a04"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "revoked_session",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("expires_at", sa.BigInteger(), nullable=True),
    )
    op.create_index("ix_revoked_session_expires_at", "revoked_session", ["expires_at"])
    op.create_table(
        "security_artifact",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("scope", sa.String(64), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("local_file_id", sa.Text(), nullable=True),
        sa.Column("chat_id", sa.Text(), nullable=True),
        sa.Column("model_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
    )
    op.create_index("ix_security_artifact_user_id", "security_artifact", ["user_id"])


def downgrade():
    op.drop_table("security_artifact")
    op.drop_table("revoked_session")

"""Track first-party session lineages for rotating sliding-session refresh."""

import sqlalchemy as sa
from alembic import op

revision = "2e25f6e41961"
down_revision = "a51d9012c822"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "session_lineage",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("session_stamp", sa.Text(), nullable=False),
        sa.Column("auth_method", sa.String(32), nullable=False),
        sa.Column("auth_time", sa.BigInteger(), nullable=False),
        sa.Column("mfa_verified", sa.Boolean(), nullable=False),
        sa.Column("absolute_expires_at", sa.BigInteger(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("current_secret", sa.String(64), nullable=False),
        sa.Column("previous_secret", sa.String(64), nullable=True),
        sa.Column("current_jti", sa.String(64), nullable=False),
        sa.Column("previous_jti", sa.String(64), nullable=True),
        sa.Column("rotated_at", sa.BigInteger(), nullable=False),
        sa.Column("oauth_session_id", sa.Text(), nullable=True),
        sa.Column("revoked_at", sa.BigInteger(), nullable=True),
    )
    op.create_index("ix_session_lineage_user_id", "session_lineage", ["user_id"])
    op.create_index(
        "ix_session_lineage_absolute_expires_at",
        "session_lineage",
        ["absolute_expires_at"],
    )


def downgrade():
    op.drop_index("ix_session_lineage_absolute_expires_at", table_name="session_lineage")
    op.drop_index("ix_session_lineage_user_id", table_name="session_lineage")
    op.drop_table("session_lineage")

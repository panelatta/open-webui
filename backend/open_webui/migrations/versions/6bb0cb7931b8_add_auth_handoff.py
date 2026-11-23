"""One-time OAuth sign-in hand-over tickets (single use, short lived, token encrypted at rest)."""

import sqlalchemy as sa
from alembic import op

revision = "6bb0cb7931b8"
down_revision = "2e25f6e41961"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "auth_handoff",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("auth_time", sa.BigInteger(), nullable=False),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.BigInteger(), nullable=False),
    )
    op.create_index("ix_auth_handoff_expires_at", "auth_handoff", ["expires_at"])


def downgrade():
    op.drop_index("ix_auth_handoff_expires_at", table_name="auth_handoff")
    op.drop_table("auth_handoff")

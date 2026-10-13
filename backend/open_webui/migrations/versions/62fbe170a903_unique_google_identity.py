"""Prevent a Google identity from being bound to multiple local accounts."""
import sqlalchemy as sa
from alembic import op

revision = '62fbe170a903'
down_revision = '57e314bdc902'
branch_labels = None
depends_on = None


def upgrade():
    dialect = op.get_bind().dialect.name
    expression = (
        "CAST(json_extract(oauth, '$.google.sub') AS TEXT)" if dialect == 'sqlite'
        else "(oauth::jsonb -> 'google' ->> 'sub')"
    )
    op.create_index('uq_user_google_sub', 'user', [sa.text(expression)], unique=True)


def downgrade():
    op.drop_index('uq_user_google_sub', table_name='user')

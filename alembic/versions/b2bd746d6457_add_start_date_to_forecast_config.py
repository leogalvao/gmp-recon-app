"""add_start_date_to_forecast_config

Revision ID: b2bd746d6457
Revises: 6bd83df28be8
Create Date: 2026-01-06 23:12:01.743540

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2bd746d6457'
down_revision: Union[str, Sequence[str], None] = '6bd83df28be8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add start_date column to forecast_config for manual override control."""
    # Check if column exists (SQLite doesn't support IF NOT EXISTS for ALTER)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('forecast_config')]

    if 'start_date' not in columns:
        op.add_column('forecast_config', sa.Column('start_date', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Remove start_date column from forecast_config."""
    op.drop_column('forecast_config', 'start_date')

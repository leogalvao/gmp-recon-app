"""add_schedule_date_fields

Revision ID: 0a5a83d79c42
Revises: fed2fcfcad69
Create Date: 2026-01-06 21:38:48.518130

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0a5a83d79c42'
down_revision: Union[str, Sequence[str], None] = 'fed2fcfcad69'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add date fields to schedule_activities table for time-based forecasting."""
    op.add_column('schedule_activities', sa.Column('start_date', sa.Date(), nullable=True))
    op.add_column('schedule_activities', sa.Column('finish_date', sa.Date(), nullable=True))
    op.add_column('schedule_activities', sa.Column('planned_start', sa.Date(), nullable=True))
    op.add_column('schedule_activities', sa.Column('planned_finish', sa.Date(), nullable=True))
    op.add_column('schedule_activities', sa.Column('duration_days', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove date fields from schedule_activities table."""
    op.drop_column('schedule_activities', 'duration_days')
    op.drop_column('schedule_activities', 'planned_finish')
    op.drop_column('schedule_activities', 'planned_start')
    op.drop_column('schedule_activities', 'finish_date')
    op.drop_column('schedule_activities', 'start_date')

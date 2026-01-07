"""add_p6_schedule_fields

Revision ID: 6bd83df28be8
Revises: 0a5a83d79c42
Create Date: 2026-01-06 22:10:55.698068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6bd83df28be8'
down_revision: Union[str, Sequence[str], None] = '0a5a83d79c42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add P6-specific fields to schedule_activities table."""
    # P6 actual date tracking (suffix " A" detection)
    op.add_column('schedule_activities', sa.Column('start_is_actual', sa.Boolean(), server_default='0', nullable=False))
    op.add_column('schedule_activities', sa.Column('finish_is_actual', sa.Boolean(), server_default='0', nullable=False))

    # P6 derived progress state
    op.add_column('schedule_activities', sa.Column('is_complete', sa.Boolean(), server_default='0', nullable=False))
    op.add_column('schedule_activities', sa.Column('is_in_progress', sa.Boolean(), server_default='0', nullable=False))
    op.add_column('schedule_activities', sa.Column('progress_pct', sa.Float(), server_default='0.0', nullable=False))

    # Critical path weighting
    op.add_column('schedule_activities', sa.Column('total_float', sa.Integer(), nullable=True))
    op.add_column('schedule_activities', sa.Column('is_critical', sa.Boolean(), server_default='0', nullable=False))

    # GMP mapping source tracking
    op.add_column('schedule_activities', sa.Column('mapping_source', sa.String(30), server_default='manual', nullable=False))
    op.add_column('schedule_activities', sa.Column('mapping_confidence', sa.Float(), server_default='1.0', nullable=False))


def downgrade() -> None:
    """Remove P6-specific fields from schedule_activities table."""
    op.drop_column('schedule_activities', 'mapping_confidence')
    op.drop_column('schedule_activities', 'mapping_source')
    op.drop_column('schedule_activities', 'is_critical')
    op.drop_column('schedule_activities', 'total_float')
    op.drop_column('schedule_activities', 'progress_pct')
    op.drop_column('schedule_activities', 'is_in_progress')
    op.drop_column('schedule_activities', 'is_complete')
    op.drop_column('schedule_activities', 'finish_is_actual')
    op.drop_column('schedule_activities', 'start_is_actual')

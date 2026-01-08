"""add_schedule_activity_zone

Revision ID: 20260108_sched_zone
Revises: 20260108_spatial
Create Date: 2026-01-08 14:00:00.000000

Migration for Schedule-to-GMP Zone Mapping feature.
Adds zone column to schedule_activities table for spatial linkage to GMP funding buckets.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260108_sched_zone'
down_revision: Union[str, Sequence[str], None] = '20260108_spatial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add zone column to schedule_activities table."""

    # Add zone column for spatial linkage to GMP funding bucket
    op.add_column(
        'schedule_activities',
        sa.Column('zone', sa.String(length=10), nullable=True)
    )

    # Create index on (project-scoped filtering not available, but zone index helps)
    op.create_index(
        'ix_schedule_activities_zone',
        'schedule_activities',
        ['zone'],
        unique=False
    )


def downgrade() -> None:
    """Remove zone column from schedule_activities table."""

    op.drop_index('ix_schedule_activities_zone', table_name='schedule_activities')
    op.drop_column('schedule_activities', 'zone')

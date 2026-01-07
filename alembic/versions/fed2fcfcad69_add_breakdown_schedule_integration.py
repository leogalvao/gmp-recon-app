"""add_breakdown_schedule_integration

Revision ID: fed2fcfcad69
Revises: 20260105_side
Create Date: 2026-01-06 20:34:42.699640

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fed2fcfcad69'
down_revision: Union[str, Sequence[str], None] = '20260105_side'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new columns to settings table
    op.add_column('settings', sa.Column('use_breakdown_allocations', sa.Boolean(), nullable=True, server_default='1'))
    op.add_column('settings', sa.Column('use_schedule_forecast', sa.Boolean(), nullable=True, server_default='0'))

    # Create new tables for breakdown and schedule integration
    op.create_table('gmp_budget_breakdown',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cost_code_description', sa.String(200), nullable=False),
        sa.Column('gmp_division', sa.String(200), nullable=True),
        sa.Column('gmp_sov_cents', sa.Integer(), nullable=False),
        sa.Column('east_funded_cents', sa.Integer(), server_default='0'),
        sa.Column('west_funded_cents', sa.Integer(), server_default='0'),
        sa.Column('pct_east', sa.Float(), nullable=False),
        sa.Column('pct_west', sa.Float(), nullable=False),
        sa.Column('match_score', sa.Integer(), server_default='0'),
        sa.Column('source_file', sa.String(100), nullable=True),
        sa.Column('imported_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_gmp_budget_breakdown_gmp_division', 'gmp_budget_breakdown', ['gmp_division'])

    op.create_table('schedule_activities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('row_number', sa.Integer(), nullable=True),
        sa.Column('task_name', sa.String(500), nullable=False),
        sa.Column('source_uid', sa.String(100), nullable=True, unique=True),
        sa.Column('activity_id', sa.String(50), nullable=True),
        sa.Column('wbs', sa.String(100), nullable=True),
        sa.Column('pct_complete', sa.Integer(), server_default='0'),
        sa.Column('imported_at', sa.DateTime(), nullable=True),
        sa.Column('source_file', sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_schedule_activities_activity_id', 'schedule_activities', ['activity_id'])
    op.create_index('ix_schedule_activities_wbs', 'schedule_activities', ['wbs'])

    op.create_table('schedule_to_gmp_mapping',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('schedule_activity_id', sa.Integer(), nullable=False),
        sa.Column('gmp_division', sa.String(200), nullable=False),
        sa.Column('weight', sa.Float(), server_default='1.0'),
        sa.Column('created_by', sa.String(100), server_default="'system'"),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['schedule_activity_id'], ['schedule_activities.id']),
        sa.UniqueConstraint('schedule_activity_id', 'gmp_division', name='uq_schedule_gmp')
    )
    op.create_index('ix_schedule_to_gmp_mapping_gmp_division', 'schedule_to_gmp_mapping', ['gmp_division'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop new tables
    op.drop_index('ix_schedule_to_gmp_mapping_gmp_division', table_name='schedule_to_gmp_mapping')
    op.drop_table('schedule_to_gmp_mapping')
    op.drop_index('ix_schedule_activities_wbs', table_name='schedule_activities')
    op.drop_index('ix_schedule_activities_activity_id', table_name='schedule_activities')
    op.drop_table('schedule_activities')
    op.drop_index('ix_gmp_budget_breakdown_gmp_division', table_name='gmp_budget_breakdown')
    op.drop_table('gmp_budget_breakdown')

    # Drop new columns from settings
    op.drop_column('settings', 'use_schedule_forecast')
    op.drop_column('settings', 'use_breakdown_allocations')

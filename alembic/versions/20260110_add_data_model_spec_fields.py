"""add_data_model_spec_fields

Revision ID: 20260110_spec_v211
Revises: 20260108_sched_zone
Create Date: 2026-01-10 10:00:00.000000

Migration for Data Model Spec v2.1.1 compliance.
Adds:
- GMP: is_hard_cost field for hard/soft cost classification
- BudgetEntity: budget_code_base (derived key), direct_cost_total_cents, cost_type, sub_job
- GMPBudgetBreakdown: division_id FK for proper relationship to GMP
- ScheduleToGMPMapping: zone field for spatial allocation
- ScheduleActivity: activity_name field (spec-compliant field name)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260110_spec_v211'
down_revision: Union[str, Sequence[str], None] = '20260108_sched_zone'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add fields for Data Model Spec v2.1.1 compliance."""

    # =========================================================================
    # 1. GMP Entity: Add is_hard_cost field
    # =========================================================================
    op.add_column(
        'gmp_entities',
        sa.Column('is_hard_cost', sa.Boolean(), nullable=True, server_default='1')
    )
    op.create_index(
        'ix_gmp_entities_is_hard_cost',
        'gmp_entities',
        ['is_hard_cost'],
        unique=False
    )

    # =========================================================================
    # 2. BudgetEntity: Add budget_code_base, direct_cost_total_cents, cost_type, sub_job
    # =========================================================================
    op.add_column(
        'budget_entities',
        sa.Column('budget_code_base', sa.String(length=50), nullable=True)
    )
    op.create_index(
        'ix_budget_entities_budget_code_base',
        'budget_entities',
        ['budget_code_base'],
        unique=False
    )

    op.add_column(
        'budget_entities',
        sa.Column('direct_cost_total_cents', sa.Integer(), nullable=True, server_default='0')
    )

    op.add_column(
        'budget_entities',
        sa.Column('cost_type', sa.String(length=20), nullable=True)
    )
    op.create_index(
        'ix_budget_entities_cost_type',
        'budget_entities',
        ['cost_type'],
        unique=False
    )

    op.add_column(
        'budget_entities',
        sa.Column('sub_job', sa.String(length=50), nullable=True)
    )
    op.create_index(
        'ix_budget_entities_sub_job',
        'budget_entities',
        ['sub_job'],
        unique=False
    )

    # =========================================================================
    # 3. GMPBudgetBreakdown: Add division_id FK
    # =========================================================================
    op.add_column(
        'gmp_budget_breakdown',
        sa.Column('division_id', sa.Integer(), nullable=True)
    )
    op.create_index(
        'ix_gmp_budget_breakdown_division_id',
        'gmp_budget_breakdown',
        ['division_id'],
        unique=False
    )
    # Note: SQLite doesn't support adding FK constraints after table creation
    # The FK relationship is handled at ORM level

    # =========================================================================
    # 4. ScheduleToGMPMapping: Add zone field
    # =========================================================================
    op.add_column(
        'schedule_to_gmp_mapping',
        sa.Column('zone', sa.String(length=10), nullable=True)
    )
    op.create_index(
        'ix_schedule_to_gmp_mapping_zone',
        'schedule_to_gmp_mapping',
        ['zone'],
        unique=False
    )

    # =========================================================================
    # 5. ScheduleActivity: Add activity_name field
    # =========================================================================
    op.add_column(
        'schedule_activities',
        sa.Column('activity_name', sa.String(length=500), nullable=True)
    )

    # Backfill activity_name from task_name for existing records
    op.execute("UPDATE schedule_activities SET activity_name = task_name WHERE activity_name IS NULL")


def downgrade() -> None:
    """Remove fields added for Data Model Spec v2.1.1."""

    # ScheduleActivity: Remove activity_name
    op.drop_column('schedule_activities', 'activity_name')

    # ScheduleToGMPMapping: Remove zone
    op.drop_index('ix_schedule_to_gmp_mapping_zone', table_name='schedule_to_gmp_mapping')
    op.drop_column('schedule_to_gmp_mapping', 'zone')

    # GMPBudgetBreakdown: Remove division_id
    op.drop_index('ix_gmp_budget_breakdown_division_id', table_name='gmp_budget_breakdown')
    op.drop_column('gmp_budget_breakdown', 'division_id')

    # BudgetEntity: Remove new fields
    op.drop_index('ix_budget_entities_sub_job', table_name='budget_entities')
    op.drop_column('budget_entities', 'sub_job')
    op.drop_index('ix_budget_entities_cost_type', table_name='budget_entities')
    op.drop_column('budget_entities', 'cost_type')
    op.drop_column('budget_entities', 'direct_cost_total_cents')
    op.drop_index('ix_budget_entities_budget_code_base', table_name='budget_entities')
    op.drop_column('budget_entities', 'budget_code_base')

    # GMP Entity: Remove is_hard_cost
    op.drop_index('ix_gmp_entities_is_hard_cost', table_name='gmp_entities')
    op.drop_column('gmp_entities', 'is_hard_cost')

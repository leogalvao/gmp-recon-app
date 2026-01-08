"""add_spatial_learning_entities

Revision ID: 20260108_spatial
Revises: 6bd83df28be8
Create Date: 2026-01-08 12:00:00.000000

Migration for Spatial & Active Learning Entities per Specification v5.0.
Adds:
- Project: Top-level project entity
- GMP (gmp_entities): Funding ceiling split by Division AND Zone
- BudgetEntity: Allocated bucket of money with zone assignment
- ChangeOrder: Contract modifications (ONLY way to adjust GMP ceiling)
- DirectCostEntity: Actual transactions with retainage tracking
- TrainingRound: ML engine version tracking
- TrainingForecastSnapshot: Forecast curve points per training round
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260108_spatial'
down_revision: Union[str, Sequence[str], None] = '6bd83df28be8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # =========================================================================
    # 1. Create projects table (Level 0 in hierarchy)
    # =========================================================================
    op.create_table(
        'projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('version_id', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_projects_id', 'projects', ['id'], unique=False)
    op.create_index('ix_projects_uuid', 'projects', ['uuid'], unique=True)
    op.create_index('ix_projects_code', 'projects', ['code'], unique=True)
    op.create_index('ix_projects_is_active', 'projects', ['is_active'], unique=False)

    # =========================================================================
    # 2. Create gmp_entities table (Level 1 - Funding Ceiling)
    # =========================================================================
    op.create_table(
        'gmp_entities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('division', sa.String(length=200), nullable=False),
        sa.Column('zone', sa.String(length=10), nullable=False),  # EAST, WEST, SHARED
        sa.Column('original_amount_cents', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version_id', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'division', 'zone', name='uq_project_division_zone')
    )
    op.create_index('ix_gmp_entities_id', 'gmp_entities', ['id'], unique=False)
    op.create_index('ix_gmp_entities_uuid', 'gmp_entities', ['uuid'], unique=True)
    op.create_index('ix_gmp_entities_project_id', 'gmp_entities', ['project_id'], unique=False)
    op.create_index('ix_gmp_entities_division', 'gmp_entities', ['division'], unique=False)
    op.create_index('ix_gmp_entities_zone', 'gmp_entities', ['zone'], unique=False)

    # =========================================================================
    # 3. Create budget_entities table (Level 2 - The Plan)
    # =========================================================================
    op.create_table(
        'budget_entities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=False),
        sa.Column('gmp_id', sa.Integer(), nullable=False),
        sa.Column('cost_code', sa.String(length=50), nullable=False),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('zone', sa.String(length=10), nullable=True),  # Nullable on ingestion
        sa.Column('current_budget_cents', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('committed_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('version_id', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['gmp_id'], ['gmp_entities.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_budget_entities_id', 'budget_entities', ['id'], unique=False)
    op.create_index('ix_budget_entities_uuid', 'budget_entities', ['uuid'], unique=True)
    op.create_index('ix_budget_entities_gmp_id', 'budget_entities', ['gmp_id'], unique=False)
    op.create_index('ix_budget_entities_cost_code', 'budget_entities', ['cost_code'], unique=False)
    op.create_index('ix_budget_entities_zone', 'budget_entities', ['zone'], unique=False)

    # =========================================================================
    # 4. Create change_orders table (Level 3b - Contract Modifications)
    # =========================================================================
    op.create_table(
        'change_orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=False),
        sa.Column('gmp_id', sa.Integer(), nullable=False),
        sa.Column('number', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='draft'),
        sa.Column('amount_cents', sa.Integer(), nullable=False),
        sa.Column('requested_date', sa.Date(), nullable=True),
        sa.Column('approved_date', sa.Date(), nullable=True),
        sa.Column('approved_by', sa.String(length=100), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('version_id', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['gmp_id'], ['gmp_entities.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('gmp_id', 'number', name='uq_gmp_co_number')
    )
    op.create_index('ix_change_orders_id', 'change_orders', ['id'], unique=False)
    op.create_index('ix_change_orders_uuid', 'change_orders', ['uuid'], unique=True)
    op.create_index('ix_change_orders_gmp_id', 'change_orders', ['gmp_id'], unique=False)
    op.create_index('ix_change_orders_number', 'change_orders', ['number'], unique=False)

    # =========================================================================
    # 5. Create direct_cost_entities table (Level 3 - The Actuals)
    # =========================================================================
    op.create_table(
        'direct_cost_entities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=False),
        sa.Column('mapped_budget_id', sa.Integer(), nullable=True),
        sa.Column('source_row_id', sa.Integer(), nullable=True),
        sa.Column('vendor_name', sa.String(length=255), nullable=True),
        sa.Column('vendor_normalized', sa.String(length=255), nullable=True),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('transaction_date', sa.Date(), nullable=True),
        sa.Column('gross_amount_cents', sa.Integer(), nullable=False),
        sa.Column('retainage_amount_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('allocation_method', sa.String(length=20), nullable=True, server_default='direct'),
        sa.Column('zone', sa.String(length=10), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['mapped_budget_id'], ['budget_entities.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_direct_cost_entities_id', 'direct_cost_entities', ['id'], unique=False)
    op.create_index('ix_direct_cost_entities_uuid', 'direct_cost_entities', ['uuid'], unique=True)
    op.create_index('ix_direct_cost_entities_mapped_budget_id', 'direct_cost_entities', ['mapped_budget_id'], unique=False)
    op.create_index('ix_direct_cost_entities_vendor_normalized', 'direct_cost_entities', ['vendor_normalized'], unique=False)
    op.create_index('ix_direct_cost_entities_transaction_date', 'direct_cost_entities', ['transaction_date'], unique=False)
    op.create_index('ix_direct_cost_entities_zone', 'direct_cost_entities', ['zone'], unique=False)

    # =========================================================================
    # 6. Create training_rounds table (The "Brain" Versioning)
    # =========================================================================
    op.create_table(
        'training_rounds',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('triggered_at', sa.DateTime(), nullable=False),
        sa.Column('trigger_type', sa.String(length=50), nullable=True, server_default='manual'),
        sa.Column('status', sa.String(length=20), nullable=True, server_default='pending'),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        # Performance metrics
        sa.Column('linkage_score', sa.Float(), nullable=True),
        sa.Column('mapping_accuracy', sa.Float(), nullable=True),
        sa.Column('budget_coverage', sa.Float(), nullable=True),
        sa.Column('cost_coverage', sa.Float(), nullable=True),
        # Model metadata
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('model_params', sa.Text(), nullable=True),
        sa.Column('training_notes', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        # Comparison with previous round
        sa.Column('previous_round_id', sa.Integer(), nullable=True),
        sa.Column('eac_change_cents', sa.Integer(), nullable=True),
        sa.Column('eac_change_pct', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_training_rounds_id', 'training_rounds', ['id'], unique=False)
    op.create_index('ix_training_rounds_uuid', 'training_rounds', ['uuid'], unique=True)
    op.create_index('ix_training_rounds_project_id', 'training_rounds', ['project_id'], unique=False)
    op.create_index('ix_training_rounds_triggered_at', 'training_rounds', ['triggered_at'], unique=False)

    # =========================================================================
    # 7. Create training_forecast_snapshots table (Forecast Curve Points)
    # =========================================================================
    op.create_table(
        'training_forecast_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('training_round_id', sa.Integer(), nullable=False),
        sa.Column('period_date', sa.Date(), nullable=False),
        sa.Column('predicted_cumulative_cost_cents', sa.Integer(), nullable=False),
        sa.Column('actual_cumulative_cost_cents', sa.Integer(), nullable=True),
        sa.Column('zone', sa.String(length=10), nullable=False),  # EAST, WEST, SHARED
        sa.Column('confidence_lower_cents', sa.Integer(), nullable=True),
        sa.Column('confidence_upper_cents', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['training_round_id'], ['training_rounds.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('training_round_id', 'period_date', 'zone', name='uq_training_period_zone')
    )
    op.create_index('ix_training_forecast_snapshots_id', 'training_forecast_snapshots', ['id'], unique=False)
    op.create_index('ix_training_forecast_snapshots_training_round_id', 'training_forecast_snapshots', ['training_round_id'], unique=False)
    op.create_index('ix_training_forecast_snapshots_period_date', 'training_forecast_snapshots', ['period_date'], unique=False)
    op.create_index('ix_training_forecast_snapshots_zone', 'training_forecast_snapshots', ['zone'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""

    # Drop training_forecast_snapshots
    op.drop_index('ix_training_forecast_snapshots_zone', table_name='training_forecast_snapshots')
    op.drop_index('ix_training_forecast_snapshots_period_date', table_name='training_forecast_snapshots')
    op.drop_index('ix_training_forecast_snapshots_training_round_id', table_name='training_forecast_snapshots')
    op.drop_index('ix_training_forecast_snapshots_id', table_name='training_forecast_snapshots')
    op.drop_table('training_forecast_snapshots')

    # Drop training_rounds
    op.drop_index('ix_training_rounds_triggered_at', table_name='training_rounds')
    op.drop_index('ix_training_rounds_project_id', table_name='training_rounds')
    op.drop_index('ix_training_rounds_uuid', table_name='training_rounds')
    op.drop_index('ix_training_rounds_id', table_name='training_rounds')
    op.drop_table('training_rounds')

    # Drop direct_cost_entities
    op.drop_index('ix_direct_cost_entities_zone', table_name='direct_cost_entities')
    op.drop_index('ix_direct_cost_entities_transaction_date', table_name='direct_cost_entities')
    op.drop_index('ix_direct_cost_entities_vendor_normalized', table_name='direct_cost_entities')
    op.drop_index('ix_direct_cost_entities_mapped_budget_id', table_name='direct_cost_entities')
    op.drop_index('ix_direct_cost_entities_uuid', table_name='direct_cost_entities')
    op.drop_index('ix_direct_cost_entities_id', table_name='direct_cost_entities')
    op.drop_table('direct_cost_entities')

    # Drop change_orders
    op.drop_index('ix_change_orders_number', table_name='change_orders')
    op.drop_index('ix_change_orders_gmp_id', table_name='change_orders')
    op.drop_index('ix_change_orders_uuid', table_name='change_orders')
    op.drop_index('ix_change_orders_id', table_name='change_orders')
    op.drop_table('change_orders')

    # Drop budget_entities
    op.drop_index('ix_budget_entities_zone', table_name='budget_entities')
    op.drop_index('ix_budget_entities_cost_code', table_name='budget_entities')
    op.drop_index('ix_budget_entities_gmp_id', table_name='budget_entities')
    op.drop_index('ix_budget_entities_uuid', table_name='budget_entities')
    op.drop_index('ix_budget_entities_id', table_name='budget_entities')
    op.drop_table('budget_entities')

    # Drop gmp_entities
    op.drop_index('ix_gmp_entities_zone', table_name='gmp_entities')
    op.drop_index('ix_gmp_entities_division', table_name='gmp_entities')
    op.drop_index('ix_gmp_entities_project_id', table_name='gmp_entities')
    op.drop_index('ix_gmp_entities_uuid', table_name='gmp_entities')
    op.drop_index('ix_gmp_entities_id', table_name='gmp_entities')
    op.drop_table('gmp_entities')

    # Drop projects
    op.drop_index('ix_projects_is_active', table_name='projects')
    op.drop_index('ix_projects_code', table_name='projects')
    op.drop_index('ix_projects_uuid', table_name='projects')
    op.drop_index('ix_projects_id', table_name='projects')
    op.drop_table('projects')

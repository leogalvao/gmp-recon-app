"""add_forecasting_tables

Revision ID: 20260104_forecast
Revises: 10bfefa143f1
Create Date: 2026-01-04 12:00:00.000000

Migration for GMP Line-Level Forecasting Module.
Adds:
- ForecastConfig: method selection and parameters per GMP division
- ForecastSnapshot: point-in-time forecast with EAC, CPI, confidence
- ForecastPeriod: time-bucketed forecasts (weekly/monthly)
- ForecastAuditLog: audit trail for forecast changes
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260104_forecast'
down_revision: Union[str, Sequence[str], None] = '10bfefa143f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # =========================================================================
    # 1. Create forecast_config table
    # =========================================================================
    op.create_table(
        'forecast_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('gmp_division', sa.String(length=200), nullable=False),
        sa.Column('method', sa.String(length=30), nullable=True, server_default='evm'),
        # EVM parameters
        sa.Column('evm_performance_factor', sa.Float(), nullable=True, server_default='1.0'),
        # PERT parameters (in cents)
        sa.Column('pert_optimistic_cents', sa.Integer(), nullable=True),
        sa.Column('pert_most_likely_cents', sa.Integer(), nullable=True),
        sa.Column('pert_pessimistic_cents', sa.Integer(), nullable=True),
        # Parametric parameters
        sa.Column('param_quantity', sa.Float(), nullable=True),
        sa.Column('param_unit_rate_cents', sa.Integer(), nullable=True),
        sa.Column('param_complexity_factor', sa.Float(), nullable=True, server_default='1.0'),
        # Distribution and completion
        sa.Column('distribution_method', sa.String(length=20), nullable=True, server_default='linear'),
        sa.Column('completion_date', sa.DateTime(), nullable=True),
        # Metadata
        sa.Column('is_locked', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True, server_default='system'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_forecast_config_id', 'forecast_config', ['id'], unique=False)
    op.create_index('ix_forecast_config_gmp_division', 'forecast_config', ['gmp_division'], unique=True)

    # =========================================================================
    # 2. Create forecast_snapshots table
    # =========================================================================
    op.create_table(
        'forecast_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('gmp_division', sa.String(length=200), nullable=False),
        sa.Column('snapshot_date', sa.DateTime(), nullable=True),
        # Forecast values (in cents)
        sa.Column('bac_cents', sa.Integer(), nullable=False),
        sa.Column('ac_cents', sa.Integer(), nullable=False),
        sa.Column('ev_cents', sa.Integer(), nullable=True),
        sa.Column('eac_cents', sa.Integer(), nullable=False),
        sa.Column('eac_west_cents', sa.Integer(), nullable=False),
        sa.Column('eac_east_cents', sa.Integer(), nullable=False),
        sa.Column('etc_cents', sa.Integer(), nullable=False),
        sa.Column('var_cents', sa.Integer(), nullable=False),
        # Performance indices
        sa.Column('cpi', sa.Float(), nullable=True),
        sa.Column('spi', sa.Float(), nullable=True),
        # Method and confidence
        sa.Column('method', sa.String(length=30), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True, server_default='0.5'),
        sa.Column('confidence_band', sa.String(length=20), nullable=True, server_default='medium'),
        sa.Column('explanation', sa.Text(), nullable=True),
        # Snapshot lifecycle
        sa.Column('is_current', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('superseded_by_id', sa.Integer(), nullable=True),
        sa.Column('trigger', sa.String(length=50), nullable=True, server_default='manual'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_forecast_snapshots_id', 'forecast_snapshots', ['id'], unique=False)
    op.create_index('ix_forecast_snapshots_gmp_division', 'forecast_snapshots', ['gmp_division'], unique=False)
    op.create_index('ix_forecast_snapshots_snapshot_date', 'forecast_snapshots', ['snapshot_date'], unique=False)
    op.create_index('ix_forecast_snapshots_is_current', 'forecast_snapshots', ['is_current'], unique=False)
    # Composite index for fast current forecast lookup
    op.create_index(
        'ix_forecast_snapshots_division_current',
        'forecast_snapshots',
        ['gmp_division', 'is_current'],
        unique=False
    )

    # =========================================================================
    # 3. Create forecast_periods table
    # =========================================================================
    op.create_table(
        'forecast_periods',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('snapshot_id', sa.Integer(), nullable=False),
        sa.Column('gmp_division', sa.String(length=200), nullable=False),
        # Period identification
        sa.Column('granularity', sa.String(length=10), nullable=False),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('period_label', sa.String(length=20), nullable=False),
        sa.Column('period_number', sa.Integer(), nullable=False),
        sa.Column('iso_week', sa.Integer(), nullable=True),
        sa.Column('iso_year', sa.Integer(), nullable=True),
        sa.Column('period_type', sa.String(length=20), nullable=False),
        # Amounts (in cents)
        sa.Column('actual_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('forecast_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('blended_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('cumulative_cents', sa.Integer(), nullable=True, server_default='0'),
        # Regional split
        sa.Column('actual_west_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('actual_east_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('forecast_west_cents', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('forecast_east_cents', sa.Integer(), nullable=True, server_default='0'),
        # Span allocation
        sa.Column('span_allocation_factor', sa.Float(), nullable=True, server_default='1.0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_forecast_periods_id', 'forecast_periods', ['id'], unique=False)
    op.create_index('ix_forecast_periods_snapshot_id', 'forecast_periods', ['snapshot_id'], unique=False)
    op.create_index('ix_forecast_periods_gmp_division', 'forecast_periods', ['gmp_division'], unique=False)
    op.create_index('ix_forecast_periods_period_start', 'forecast_periods', ['period_start'], unique=False)
    # Composite index for fast period lookup
    op.create_index(
        'ix_forecast_periods_granularity_division',
        'forecast_periods',
        ['granularity', 'gmp_division'],
        unique=False
    )

    # =========================================================================
    # 4. Create forecast_audit_log table
    # =========================================================================
    op.create_table(
        'forecast_audit_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('gmp_division', sa.String(length=200), nullable=False),
        sa.Column('action', sa.String(length=30), nullable=False),
        sa.Column('field_changed', sa.String(length=50), nullable=True),
        sa.Column('old_value', sa.Text(), nullable=True),
        sa.Column('new_value', sa.Text(), nullable=True),
        sa.Column('previous_eac_cents', sa.Integer(), nullable=True),
        sa.Column('new_eac_cents', sa.Integer(), nullable=True),
        sa.Column('change_reason', sa.String(length=200), nullable=True),
        sa.Column('changed_by', sa.String(length=100), nullable=True, server_default='system'),
        sa.Column('changed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_forecast_audit_log_id', 'forecast_audit_log', ['id'], unique=False)
    op.create_index('ix_forecast_audit_log_gmp_division', 'forecast_audit_log', ['gmp_division'], unique=False)
    op.create_index('ix_forecast_audit_log_changed_at', 'forecast_audit_log', ['changed_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""

    # Drop forecast_audit_log
    op.drop_index('ix_forecast_audit_log_changed_at', table_name='forecast_audit_log')
    op.drop_index('ix_forecast_audit_log_gmp_division', table_name='forecast_audit_log')
    op.drop_index('ix_forecast_audit_log_id', table_name='forecast_audit_log')
    op.drop_table('forecast_audit_log')

    # Drop forecast_periods
    op.drop_index('ix_forecast_periods_granularity_division', table_name='forecast_periods')
    op.drop_index('ix_forecast_periods_period_start', table_name='forecast_periods')
    op.drop_index('ix_forecast_periods_gmp_division', table_name='forecast_periods')
    op.drop_index('ix_forecast_periods_snapshot_id', table_name='forecast_periods')
    op.drop_index('ix_forecast_periods_id', table_name='forecast_periods')
    op.drop_table('forecast_periods')

    # Drop forecast_snapshots
    op.drop_index('ix_forecast_snapshots_division_current', table_name='forecast_snapshots')
    op.drop_index('ix_forecast_snapshots_is_current', table_name='forecast_snapshots')
    op.drop_index('ix_forecast_snapshots_snapshot_date', table_name='forecast_snapshots')
    op.drop_index('ix_forecast_snapshots_gmp_division', table_name='forecast_snapshots')
    op.drop_index('ix_forecast_snapshots_id', table_name='forecast_snapshots')
    op.drop_table('forecast_snapshots')

    # Drop forecast_config
    op.drop_index('ix_forecast_config_gmp_division', table_name='forecast_config')
    op.drop_index('ix_forecast_config_id', table_name='forecast_config')
    op.drop_table('forecast_config')

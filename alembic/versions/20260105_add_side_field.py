"""add_side_field

Revision ID: 20260105_side
Revises: 20260104_forecast
Create Date: 2026-01-05 12:00:00.000000

Migration for Side Assignment (East/West/Both) feature.
Adds:
- side column to budget_to_gmp table (default: BOTH)
- side column to direct_to_budget table (default: BOTH)
- side_configuration table for timeline and allocation settings
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260105_side'
down_revision: Union[str, Sequence[str], None] = '20260104_forecast'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # =========================================================================
    # 1. Add side column to budget_to_gmp table
    # =========================================================================
    op.add_column('budget_to_gmp', sa.Column(
        'side', sa.String(length=4), nullable=False, server_default='BOTH'
    ))
    op.create_index(
        'ix_budget_to_gmp_side',
        'budget_to_gmp',
        ['side'],
        unique=False
    )
    # Composite index for filtering by division + side
    op.create_index(
        'ix_budget_to_gmp_division_side',
        'budget_to_gmp',
        ['gmp_division', 'side'],
        unique=False
    )

    # =========================================================================
    # 2. Add side column to direct_to_budget table
    # =========================================================================
    op.add_column('direct_to_budget', sa.Column(
        'side', sa.String(length=4), nullable=False, server_default='BOTH'
    ))
    op.create_index(
        'ix_direct_to_budget_side',
        'direct_to_budget',
        ['side'],
        unique=False
    )
    # Composite index for filtering by budget_code + side
    op.create_index(
        'ix_direct_to_budget_budget_side',
        'direct_to_budget',
        ['budget_code', 'side'],
        unique=False
    )

    # =========================================================================
    # 3. Create side_configuration table
    # =========================================================================
    op.create_table(
        'side_configuration',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('side', sa.String(length=4), nullable=False),
        sa.Column('display_name', sa.String(length=20), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('allocation_weight', sa.Float(), nullable=True, server_default='0.5'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('side', name='uq_side_configuration_side')
    )
    op.create_index('ix_side_configuration_id', 'side_configuration', ['id'], unique=False)
    op.create_index('ix_side_configuration_side', 'side_configuration', ['side'], unique=True)
    op.create_index('ix_side_configuration_is_active', 'side_configuration', ['is_active'], unique=False)

    # =========================================================================
    # 4. Seed default side configuration data
    # =========================================================================
    # Insert default configuration for East, West, Both
    # East: ends July 31, 2025
    # West: starts June 1, 2025, ends July 31, 2026
    # Both: always active, full allocation weight
    op.execute("""
        INSERT INTO side_configuration (side, display_name, start_date, end_date, is_active, allocation_weight, created_at, updated_at)
        VALUES
            ('EAST', 'East', NULL, '2025-07-31 23:59:59', 1, 0.5, datetime('now'), datetime('now')),
            ('WEST', 'West', '2025-06-01 00:00:00', '2026-07-31 23:59:59', 1, 0.5, datetime('now'), datetime('now')),
            ('BOTH', 'Both', NULL, NULL, 1, 1.0, datetime('now'), datetime('now'))
    """)


def downgrade() -> None:
    """Downgrade schema."""

    # Drop side_configuration table
    op.drop_index('ix_side_configuration_is_active', table_name='side_configuration')
    op.drop_index('ix_side_configuration_side', table_name='side_configuration')
    op.drop_index('ix_side_configuration_id', table_name='side_configuration')
    op.drop_table('side_configuration')

    # Remove side column from direct_to_budget
    op.drop_index('ix_direct_to_budget_budget_side', table_name='direct_to_budget')
    op.drop_index('ix_direct_to_budget_side', table_name='direct_to_budget')
    op.drop_column('direct_to_budget', 'side')

    # Remove side column from budget_to_gmp
    op.drop_index('ix_budget_to_gmp_division_side', table_name='budget_to_gmp')
    op.drop_index('ix_budget_to_gmp_side', table_name='budget_to_gmp')
    op.drop_column('budget_to_gmp', 'side')

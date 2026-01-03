"""add_enhanced_mapping_tables

Revision ID: 10bfefa143f1
Revises:
Create Date: 2026-01-03 07:47:47.850442

Migration for Enhanced Direct Cost → Budget Mapping.
Adds:
- MappingFeedback: stores learned patterns from user interactions
- BudgetMatchStats: aggregated stats for historical lookup
- SuggestionCache: precomputed match suggestions
- New columns on DirectToBudget: method, vendor_normalized
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '10bfefa143f1'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # =========================================================================
    # 1. Add columns to existing direct_to_budget table
    # =========================================================================
    op.add_column('direct_to_budget', sa.Column(
        'method', sa.String(length=30), nullable=True, server_default='manual'
    ))
    op.add_column('direct_to_budget', sa.Column(
        'vendor_normalized', sa.String(length=255), nullable=True
    ))
    op.create_index(
        'ix_direct_to_budget_vendor_normalized',
        'direct_to_budget',
        ['vendor_normalized'],
        unique=False
    )

    # =========================================================================
    # 2. Create mapping_feedback table
    # =========================================================================
    op.create_table(
        'mapping_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('vendor_normalized', sa.String(length=255), nullable=False),
        sa.Column('name_prefix', sa.String(length=50), nullable=False),
        sa.Column('budget_code', sa.String(length=50), nullable=False),
        sa.Column('was_override', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('suggested_budget_code', sa.String(length=50), nullable=True),
        sa.Column('confidence_at_suggestion', sa.Float(), nullable=True),
        sa.Column('user_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_mapping_feedback_id', 'mapping_feedback', ['id'], unique=False)
    op.create_index('ix_mapping_feedback_vendor_normalized', 'mapping_feedback', ['vendor_normalized'], unique=False)
    op.create_index('ix_mapping_feedback_name_prefix', 'mapping_feedback', ['name_prefix'], unique=False)
    op.create_index('ix_mapping_feedback_budget_code', 'mapping_feedback', ['budget_code'], unique=False)
    # Composite index for fast pattern lookup
    op.create_index(
        'ix_mapping_feedback_vendor_name',
        'mapping_feedback',
        ['vendor_normalized', 'name_prefix'],
        unique=False
    )

    # =========================================================================
    # 3. Create budget_match_stats table
    # =========================================================================
    op.create_table(
        'budget_match_stats',
        sa.Column('budget_code', sa.String(length=50), nullable=False),
        sa.Column('total_matches', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('override_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('trust_score', sa.Float(), nullable=True, server_default='1.0'),
        sa.Column('last_updated', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('budget_code')
    )

    # =========================================================================
    # 4. Create suggestion_cache table
    # =========================================================================
    op.create_table(
        'suggestion_cache',
        sa.Column('direct_cost_id', sa.Integer(), nullable=False),
        sa.Column('suggestions', sa.Text(), nullable=False),
        sa.Column('top_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('computed_at', sa.DateTime(), nullable=True),
        sa.Column('stale', sa.Boolean(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('direct_cost_id')
    )
    op.create_index('ix_suggestion_cache_top_score', 'suggestion_cache', ['top_score'], unique=False)
    op.create_index('ix_suggestion_cache_stale', 'suggestion_cache', ['stale'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""

    # Drop new tables
    op.drop_index('ix_suggestion_cache_stale', table_name='suggestion_cache')
    op.drop_index('ix_suggestion_cache_top_score', table_name='suggestion_cache')
    op.drop_table('suggestion_cache')

    op.drop_table('budget_match_stats')

    op.drop_index('ix_mapping_feedback_vendor_name', table_name='mapping_feedback')
    op.drop_index('ix_mapping_feedback_budget_code', table_name='mapping_feedback')
    op.drop_index('ix_mapping_feedback_name_prefix', table_name='mapping_feedback')
    op.drop_index('ix_mapping_feedback_vendor_normalized', table_name='mapping_feedback')
    op.drop_index('ix_mapping_feedback_id', table_name='mapping_feedback')
    op.drop_table('mapping_feedback')

    # Remove columns from direct_to_budget
    op.drop_index('ix_direct_to_budget_vendor_normalized', table_name='direct_to_budget')
    op.drop_column('direct_to_budget', 'vendor_normalized')
    op.drop_column('direct_to_budget', 'method')

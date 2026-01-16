"""Phase 1: Multi-project platform foundation

Adds tables and columns for multi-project support:
- canonical_trades: Master trade taxonomy (CSI-based)
- project_trade_mappings: Project-to-canonical trade mapping
- canonical_cost_features: Normalized cost features for ML
- project_embeddings: Project embeddings for ML
- training_datasets: Cross-project training datasets
- model_registry: Model versioning and registry
- project_forecasts: Per-project forecasts

Modifies:
- projects: Add square footage, type, region, training eligibility
- gmp_entities: Add canonical_trade_id and normalized amount

Revision ID: 20260116_phase1
Revises: 6d20668f2537
Create Date: 2026-01-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260116_phase1'
down_revision: Union[str, Sequence[str], None] = '6d20668f2537'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name):
    """Check if a table exists in the database."""
    from sqlalchemy import inspect
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name, column_name):
    """Check if a column exists in a table."""
    from sqlalchemy import inspect
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Upgrade schema for Phase 1 multi-project foundation."""

    # =========================================================================
    # 1. CANONICAL TRADES - Master Trade Taxonomy
    # =========================================================================
    if not table_exists('canonical_trades'):
        op.create_table(
            'canonical_trades',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('canonical_code', sa.String(20), nullable=False),
            sa.Column('csi_division', sa.String(2), nullable=False),
            sa.Column('canonical_name', sa.String(200), nullable=False),
            sa.Column('parent_trade_id', sa.Integer(), nullable=True),
            sa.Column('hierarchy_level', sa.Integer(), nullable=False, server_default='1'),
            sa.Column('typical_pct_of_total', sa.Float(), nullable=True),
            sa.Column('typical_duration_pct', sa.Float(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['parent_trade_id'], ['canonical_trades.id']),
        )
        op.create_index('ix_canonical_trades_canonical_code', 'canonical_trades', ['canonical_code'], unique=True)
        op.create_index('ix_canonical_trades_csi_division', 'canonical_trades', ['csi_division'], unique=False)

    # =========================================================================
    # 2. PROJECT TRADE MAPPINGS - Project-to-Canonical Trade Mapping
    # =========================================================================
    if not table_exists('project_trade_mappings'):
        op.create_table(
            'project_trade_mappings',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('project_id', sa.Integer(), nullable=False),
            sa.Column('raw_division_name', sa.String(200), nullable=False),
            sa.Column('canonical_trade_id', sa.Integer(), nullable=False),
            sa.Column('confidence', sa.Float(), nullable=False, server_default='1.0'),
            sa.Column('mapping_method', sa.String(30), nullable=False),
            sa.Column('created_by', sa.String(100), nullable=False, server_default="'system'"),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
            sa.ForeignKeyConstraint(['canonical_trade_id'], ['canonical_trades.id']),
            sa.UniqueConstraint('project_id', 'raw_division_name', name='uq_project_raw_division'),
        )
        op.create_index('ix_project_trade_mappings_project_id', 'project_trade_mappings', ['project_id'])
        op.create_index('ix_project_trade_mappings_canonical_trade_id', 'project_trade_mappings', ['canonical_trade_id'])

    # =========================================================================
    # 3. CANONICAL COST FEATURES - Normalized Cost Features for ML
    # =========================================================================
    if not table_exists('canonical_cost_features'):
        op.create_table(
            'canonical_cost_features',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('project_id', sa.Integer(), nullable=False),
            sa.Column('canonical_trade_id', sa.Integer(), nullable=False),
            sa.Column('period_date', sa.Date(), nullable=False),
            sa.Column('period_type', sa.String(10), nullable=False),
            sa.Column('cost_per_sf_cents', sa.Integer(), nullable=False),
            sa.Column('cumulative_cost_per_sf_cents', sa.Integer(), nullable=False),
            sa.Column('budget_per_sf_cents', sa.Integer(), nullable=False),
            sa.Column('pct_complete', sa.Float(), nullable=True),
            sa.Column('schedule_pct_elapsed', sa.Float(), nullable=True),
            sa.Column('pct_east', sa.Float(), nullable=False, server_default='0.5'),
            sa.Column('pct_west', sa.Float(), nullable=False, server_default='0.5'),
            sa.Column('is_backfill', sa.Boolean(), nullable=False, server_default='0'),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
            sa.ForeignKeyConstraint(['canonical_trade_id'], ['canonical_trades.id']),
        )
        op.create_index(
            'ix_canonical_cost_features_lookup',
            'canonical_cost_features',
            ['project_id', 'canonical_trade_id', 'period_date', 'period_type'],
            unique=True
        )

    # =========================================================================
    # 4. PROJECT EMBEDDINGS - Project Embeddings for ML
    # =========================================================================
    if not table_exists('project_embeddings'):
        op.create_table(
            'project_embeddings',
            sa.Column('project_id', sa.Integer(), nullable=False),
            sa.Column('embedding_vector', sa.Text(), nullable=False),
            sa.Column('model_version', sa.String(50), nullable=False),
            sa.Column('computed_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('project_id'),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
        )

    # =========================================================================
    # 5. TRAINING DATASETS - Cross-Project Training Datasets
    # =========================================================================
    if not table_exists('training_datasets'):
        op.create_table(
            'training_datasets',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('dataset_name', sa.String(100), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('project_ids', sa.Text(), nullable=False),
            sa.Column('canonical_trade_ids', sa.Text(), nullable=True),
            sa.Column('date_range_start', sa.Date(), nullable=False),
            sa.Column('date_range_end', sa.Date(), nullable=False),
            sa.Column('sample_count', sa.Integer(), nullable=False),
            sa.Column('feature_schema', sa.Text(), nullable=False),
            sa.Column('storage_path', sa.String(500), nullable=False),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_training_datasets_name', 'training_datasets', ['dataset_name'], unique=True)

    # =========================================================================
    # 6. MODEL REGISTRY - Model Versioning and Registry
    # =========================================================================
    if not table_exists('model_registry'):
        op.create_table(
            'model_registry',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('model_name', sa.String(100), nullable=False),
            sa.Column('model_version', sa.String(50), nullable=False),
            sa.Column('model_type', sa.String(50), nullable=False),
            sa.Column('scope_project_id', sa.Integer(), nullable=True),
            sa.Column('scope_canonical_trade_id', sa.Integer(), nullable=True),
            sa.Column('training_dataset_id', sa.Integer(), nullable=True),
            sa.Column('artifact_path', sa.String(500), nullable=False),
            sa.Column('metrics', sa.Text(), nullable=False),
            sa.Column('hyperparameters', sa.Text(), nullable=True),
            sa.Column('is_production', sa.Boolean(), nullable=False, server_default='0'),
            sa.Column('promoted_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['scope_project_id'], ['projects.id']),
            sa.ForeignKeyConstraint(['scope_canonical_trade_id'], ['canonical_trades.id']),
            sa.ForeignKeyConstraint(['training_dataset_id'], ['training_datasets.id']),
            sa.UniqueConstraint('model_name', 'model_version', name='uq_model_name_version'),
        )
        op.create_index('ix_model_registry_production', 'model_registry', ['model_type', 'is_production'])

    # =========================================================================
    # 7. PROJECT FORECASTS - Per-Project Forecasts
    # =========================================================================
    if not table_exists('project_forecasts'):
        op.create_table(
            'project_forecasts',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('project_id', sa.Integer(), nullable=False),
            sa.Column('canonical_trade_id', sa.Integer(), nullable=False),
            sa.Column('model_id', sa.Integer(), nullable=False),
            sa.Column('forecast_date', sa.Date(), nullable=False),
            sa.Column('horizon_months', sa.Integer(), nullable=False),
            sa.Column('predicted_eac_cents', sa.BigInteger(), nullable=False),
            sa.Column('predicted_etc_cents', sa.BigInteger(), nullable=False),
            sa.Column('confidence_lower_cents', sa.BigInteger(), nullable=True),
            sa.Column('confidence_upper_cents', sa.BigInteger(), nullable=True),
            sa.Column('confidence_level', sa.Float(), nullable=False, server_default='0.8'),
            sa.Column('eac_east_cents', sa.BigInteger(), nullable=True),
            sa.Column('eac_west_cents', sa.BigInteger(), nullable=True),
            sa.Column('is_current', sa.Boolean(), nullable=False, server_default='1'),
            sa.Column('superseded_by_id', sa.BigInteger(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
            sa.ForeignKeyConstraint(['canonical_trade_id'], ['canonical_trades.id']),
            sa.ForeignKeyConstraint(['model_id'], ['model_registry.id']),
            sa.UniqueConstraint(
                'project_id', 'canonical_trade_id', 'forecast_date', 'model_id',
                name='uq_project_trade_forecast_model'
            ),
        )
        op.create_index(
            'ix_project_forecasts_current',
            'project_forecasts',
            ['project_id', 'canonical_trade_id'],
        )

    # =========================================================================
    # 8. ADD COLUMNS TO PROJECTS TABLE
    # =========================================================================
    if table_exists('projects') and not column_exists('projects', 'total_square_feet'):
        with op.batch_alter_table('projects') as batch_op:
            batch_op.add_column(sa.Column('total_square_feet', sa.Integer(), nullable=True))
            batch_op.add_column(sa.Column('project_type', sa.String(50), nullable=True))
            batch_op.add_column(sa.Column('region', sa.String(50), nullable=True))
            batch_op.add_column(sa.Column('owner_id', sa.Integer(), nullable=True))
            batch_op.add_column(sa.Column('is_training_eligible', sa.Boolean(), nullable=False, server_default='1'))
            batch_op.add_column(sa.Column('data_quality_score', sa.Float(), nullable=True))

    # =========================================================================
    # 9. ADD COLUMNS TO GMP_ENTITIES TABLE
    # =========================================================================
    if table_exists('gmp_entities') and not column_exists('gmp_entities', 'canonical_trade_id'):
        with op.batch_alter_table('gmp_entities') as batch_op:
            batch_op.add_column(sa.Column('canonical_trade_id', sa.Integer(), nullable=True))
            batch_op.add_column(sa.Column('normalized_amount_per_sf_cents', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema - remove Phase 1 multi-project foundation."""

    # Remove columns from gmp_entities
    if table_exists('gmp_entities') and column_exists('gmp_entities', 'canonical_trade_id'):
        with op.batch_alter_table('gmp_entities') as batch_op:
            batch_op.drop_column('normalized_amount_per_sf_cents')
            batch_op.drop_column('canonical_trade_id')

    # Remove columns from projects
    if table_exists('projects') and column_exists('projects', 'total_square_feet'):
        with op.batch_alter_table('projects') as batch_op:
            batch_op.drop_column('data_quality_score')
            batch_op.drop_column('is_training_eligible')
            batch_op.drop_column('owner_id')
            batch_op.drop_column('region')
            batch_op.drop_column('project_type')
            batch_op.drop_column('total_square_feet')

    # Drop tables in reverse order (respecting FK dependencies)
    if table_exists('project_forecasts'):
        op.drop_index('ix_project_forecasts_current', table_name='project_forecasts')
        op.drop_table('project_forecasts')

    if table_exists('model_registry'):
        op.drop_index('ix_model_registry_production', table_name='model_registry')
        op.drop_table('model_registry')

    if table_exists('training_datasets'):
        op.drop_index('ix_training_datasets_name', table_name='training_datasets')
        op.drop_table('training_datasets')

    if table_exists('project_embeddings'):
        op.drop_table('project_embeddings')

    if table_exists('canonical_cost_features'):
        op.drop_index('ix_canonical_cost_features_lookup', table_name='canonical_cost_features')
        op.drop_table('canonical_cost_features')

    if table_exists('project_trade_mappings'):
        op.drop_index('ix_project_trade_mappings_canonical_trade_id', table_name='project_trade_mappings')
        op.drop_index('ix_project_trade_mappings_project_id', table_name='project_trade_mappings')
        op.drop_table('project_trade_mappings')

    if table_exists('canonical_trades'):
        op.drop_index('ix_canonical_trades_csi_division', table_name='canonical_trades')
        op.drop_index('ix_canonical_trades_canonical_code', table_name='canonical_trades')
        op.drop_table('canonical_trades')

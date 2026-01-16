"""Add feature flag persistence tables

Revision ID: 9c6bf5306d3a
Revises: 12e59a1cfaf5
Create Date: 2026-01-16 15:52:58.161155

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c6bf5306d3a'
down_revision: Union[str, Sequence[str], None] = '12e59a1cfaf5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create feature_flag_states table
    op.create_table(
        'feature_flag_states',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('flag_name', sa.String(100), nullable=False),
        sa.Column('strategy', sa.String(50), nullable=False, server_default='disabled'),
        sa.Column('percentage', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('flag_name', name='uq_flag_name'),
    )
    op.create_index('ix_feature_flag_states_flag_name', 'feature_flag_states', ['flag_name'])

    # Create feature_flag_entities table
    op.create_table(
        'feature_flag_entities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('flag_name', sa.String(100), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('flag_name', 'entity_id', name='uq_flag_entity'),
    )
    op.create_index('ix_feature_flag_entities_flag_name', 'feature_flag_entities', ['flag_name'])
    op.create_index('ix_feature_flag_entities_entity_id', 'feature_flag_entities', ['entity_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_feature_flag_entities_entity_id', 'feature_flag_entities')
    op.drop_index('ix_feature_flag_entities_flag_name', 'feature_flag_entities')
    op.drop_table('feature_flag_entities')

    op.drop_index('ix_feature_flag_states_flag_name', 'feature_flag_states')
    op.drop_table('feature_flag_states')

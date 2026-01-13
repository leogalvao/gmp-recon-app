"""Merge migration branches

Revision ID: 760ef25c116f
Revises: 20260108_sched_zone, b2bd746d6457
Create Date: 2026-01-12 22:08:24.262843

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '760ef25c116f'
down_revision: Union[str, Sequence[str], None] = ('20260108_sched_zone', 'b2bd746d6457')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass

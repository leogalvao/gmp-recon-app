"""Phase 1: Seed canonical trades with CSI divisions

Populates the canonical_trades table with standard CSI MasterFormat divisions.

Revision ID: 20260116_phase1_seed
Revises: 20260116_phase1
Create Date: 2026-01-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from datetime import datetime


# revision identifiers, used by Alembic.
revision: str = '20260116_phase1_seed'
down_revision: Union[str, Sequence[str], None] = '20260116_phase1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# CSI MasterFormat Level 1 Divisions
CSI_DIVISIONS = [
    # Division 01 - General Requirements
    {"canonical_code": "01-GENERAL", "csi_division": "01", "canonical_name": "General Requirements",
     "typical_pct_of_total": 0.08, "typical_duration_pct": 1.0},

    # Division 02 - Existing Conditions
    {"canonical_code": "02-EXISTING", "csi_division": "02", "canonical_name": "Existing Conditions",
     "typical_pct_of_total": 0.02, "typical_duration_pct": 0.15},

    # Division 03 - Concrete
    {"canonical_code": "03-CONCRETE", "csi_division": "03", "canonical_name": "Concrete",
     "typical_pct_of_total": 0.12, "typical_duration_pct": 0.25},

    # Division 04 - Masonry
    {"canonical_code": "04-MASONRY", "csi_division": "04", "canonical_name": "Masonry",
     "typical_pct_of_total": 0.04, "typical_duration_pct": 0.20},

    # Division 05 - Metals
    {"canonical_code": "05-METALS", "csi_division": "05", "canonical_name": "Metals",
     "typical_pct_of_total": 0.10, "typical_duration_pct": 0.20},

    # Division 06 - Wood, Plastics, Composites
    {"canonical_code": "06-WOOD", "csi_division": "06", "canonical_name": "Wood, Plastics, Composites",
     "typical_pct_of_total": 0.05, "typical_duration_pct": 0.25},

    # Division 07 - Thermal & Moisture Protection
    {"canonical_code": "07-THERMAL", "csi_division": "07", "canonical_name": "Thermal & Moisture Protection",
     "typical_pct_of_total": 0.06, "typical_duration_pct": 0.30},

    # Division 08 - Openings
    {"canonical_code": "08-OPENINGS", "csi_division": "08", "canonical_name": "Openings",
     "typical_pct_of_total": 0.05, "typical_duration_pct": 0.25},

    # Division 09 - Finishes
    {"canonical_code": "09-FINISHES", "csi_division": "09", "canonical_name": "Finishes",
     "typical_pct_of_total": 0.10, "typical_duration_pct": 0.35},

    # Division 10 - Specialties
    {"canonical_code": "10-SPECIALTIES", "csi_division": "10", "canonical_name": "Specialties",
     "typical_pct_of_total": 0.02, "typical_duration_pct": 0.20},

    # Division 11 - Equipment
    {"canonical_code": "11-EQUIPMENT", "csi_division": "11", "canonical_name": "Equipment",
     "typical_pct_of_total": 0.02, "typical_duration_pct": 0.15},

    # Division 12 - Furnishings
    {"canonical_code": "12-FURNISHINGS", "csi_division": "12", "canonical_name": "Furnishings",
     "typical_pct_of_total": 0.02, "typical_duration_pct": 0.10},

    # Division 13 - Special Construction
    {"canonical_code": "13-SPECIAL", "csi_division": "13", "canonical_name": "Special Construction",
     "typical_pct_of_total": 0.01, "typical_duration_pct": 0.15},

    # Division 14 - Conveying Equipment
    {"canonical_code": "14-CONVEYING", "csi_division": "14", "canonical_name": "Conveying Equipment",
     "typical_pct_of_total": 0.03, "typical_duration_pct": 0.20},

    # Division 21 - Fire Suppression
    {"canonical_code": "21-FIRE", "csi_division": "21", "canonical_name": "Fire Suppression",
     "typical_pct_of_total": 0.02, "typical_duration_pct": 0.25},

    # Division 22 - Plumbing
    {"canonical_code": "22-PLUMBING", "csi_division": "22", "canonical_name": "Plumbing",
     "typical_pct_of_total": 0.05, "typical_duration_pct": 0.40},

    # Division 23 - HVAC
    {"canonical_code": "23-HVAC", "csi_division": "23", "canonical_name": "HVAC",
     "typical_pct_of_total": 0.08, "typical_duration_pct": 0.45},

    # Division 26 - Electrical
    {"canonical_code": "26-ELECTRICAL", "csi_division": "26", "canonical_name": "Electrical",
     "typical_pct_of_total": 0.10, "typical_duration_pct": 0.50},

    # Division 27 - Communications
    {"canonical_code": "27-COMMS", "csi_division": "27", "canonical_name": "Communications",
     "typical_pct_of_total": 0.02, "typical_duration_pct": 0.30},

    # Division 28 - Electronic Safety & Security
    {"canonical_code": "28-SAFETY", "csi_division": "28", "canonical_name": "Electronic Safety & Security",
     "typical_pct_of_total": 0.01, "typical_duration_pct": 0.25},

    # Division 31 - Earthwork
    {"canonical_code": "31-EARTHWORK", "csi_division": "31", "canonical_name": "Earthwork",
     "typical_pct_of_total": 0.04, "typical_duration_pct": 0.15},

    # Division 32 - Exterior Improvements
    {"canonical_code": "32-EXTERIOR", "csi_division": "32", "canonical_name": "Exterior Improvements",
     "typical_pct_of_total": 0.03, "typical_duration_pct": 0.20},

    # Division 33 - Utilities
    {"canonical_code": "33-UTILITIES", "csi_division": "33", "canonical_name": "Utilities",
     "typical_pct_of_total": 0.03, "typical_duration_pct": 0.20},
]


def upgrade() -> None:
    """Seed canonical trades with CSI divisions."""
    # Get the canonical_trades table
    canonical_trades = sa.table(
        'canonical_trades',
        sa.column('id', sa.Integer),
        sa.column('canonical_code', sa.String),
        sa.column('csi_division', sa.String),
        sa.column('canonical_name', sa.String),
        sa.column('parent_trade_id', sa.Integer),
        sa.column('hierarchy_level', sa.Integer),
        sa.column('typical_pct_of_total', sa.Float),
        sa.column('typical_duration_pct', sa.Float),
        sa.column('is_active', sa.Boolean),
        sa.column('created_at', sa.DateTime),
        sa.column('updated_at', sa.DateTime),
    )

    # Insert all CSI divisions
    now = datetime.utcnow()
    for division in CSI_DIVISIONS:
        op.execute(
            canonical_trades.insert().values(
                canonical_code=division['canonical_code'],
                csi_division=division['csi_division'],
                canonical_name=division['canonical_name'],
                parent_trade_id=None,
                hierarchy_level=1,
                typical_pct_of_total=division['typical_pct_of_total'],
                typical_duration_pct=division['typical_duration_pct'],
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        )


def downgrade() -> None:
    """Remove seeded canonical trades."""
    canonical_trades = sa.table(
        'canonical_trades',
        sa.column('canonical_code', sa.String),
    )

    # Delete all seeded divisions
    codes = [d['canonical_code'] for d in CSI_DIVISIONS]
    op.execute(
        canonical_trades.delete().where(
            canonical_trades.c.canonical_code.in_(codes)
        )
    )

"""
Direct Cost Entity - Atomic cost unit from field/invoices.

Implements:
- Immutable value semantics
- Cost category taxonomy (labor, material, equipment, subcontract)
- Temporal attribution (incurred_date, posted_date)
"""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class CostCategory(Enum):
    """Classification of direct cost by type."""
    LABOR = "labor"
    MATERIAL = "material"
    EQUIPMENT = "equipment"
    SUBCONTRACT = "subcontract"
    OVERHEAD = "overhead"
    CONTINGENCY = "contingency"


class CostPhase(Enum):
    """Project phase when cost was incurred."""
    DESIGN = "design"
    PRECONSTRUCTION = "preconstruction"
    CONSTRUCTION = "construction"
    CLOSEOUT = "closeout"


@dataclass(frozen=True)
class DirectCost:
    """
    Immutable direct cost entry from source systems.

    Represents the atomic unit of cost data flowing through
    the Direct Cost -> Budget -> GMP pipeline.

    Attributes:
        id: Unique identifier
        amount: Cost amount in dollars (must be non-negative)
        category: Type of cost (labor, material, etc.)
        phase: Project phase when incurred
        description: Free-text description
        vendor_id: External vendor identifier
        incurred_date: Date cost was incurred
        posted_date: Date cost was posted to system
        sub_job_id: Optional sub-job assignment
        cost_code: CSI MasterFormat code for mapping
    """

    id: UUID = field(default_factory=uuid4)
    amount: Decimal = Decimal("0.00")
    category: CostCategory = CostCategory.MATERIAL
    phase: CostPhase = CostPhase.CONSTRUCTION
    description: str = ""
    vendor_id: Optional[str] = None
    incurred_date: date = field(default_factory=date.today)
    posted_date: date = field(default_factory=date.today)
    sub_job_id: Optional[UUID] = None
    cost_code: str = ""  # CSI MasterFormat code

    def __post_init__(self):
        """Validate cost amount is non-negative."""
        if self.amount < 0:
            raise ValueError("Direct cost amount cannot be negative")

    def to_budget_contribution(self, markup_factor: Decimal = Decimal("1.0")) -> Decimal:
        """
        Transform to budget line contribution with markup.

        Args:
            markup_factor: Multiplier for burden/overhead (e.g., 1.15 for 15% markup)

        Returns:
            Marked-up amount for budget contribution
        """
        return self.amount * markup_factor

    @classmethod
    def from_dict(cls, data: dict) -> 'DirectCost':
        """
        Create DirectCost from dictionary (e.g., from DataFrame row).

        Args:
            data: Dictionary with cost data

        Returns:
            DirectCost instance
        """
        return cls(
            id=UUID(data['id']) if 'id' in data else uuid4(),
            amount=Decimal(str(data.get('amount', 0))),
            category=CostCategory(data.get('category', 'material')),
            phase=CostPhase(data.get('phase', 'construction')),
            description=data.get('description', ''),
            vendor_id=data.get('vendor_id'),
            incurred_date=data.get('incurred_date', date.today()),
            posted_date=data.get('posted_date', date.today()),
            sub_job_id=UUID(data['sub_job_id']) if data.get('sub_job_id') else None,
            cost_code=data.get('cost_code', ''),
        )

"""
GMP Allocation Entity - Top-level cost container.

Represents Guaranteed Maximum Price breakdown across sub-jobs.
Implements stochastic process for fund flow modeling.
"""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from .budget_line import BudgetLine


@dataclass
class GMPLineItem:
    """
    Individual GMP line item (e.g., 'Concrete', 'Electrical').

    Aggregates multiple budget lines into a single GMP category.

    Attributes:
        id: Unique identifier
        name: Line item name (e.g., 'Division 03 - Concrete')
        description: Detailed description
        original_amount: Initial GMP amount
        current_amount: Current GMP amount (after changes)
        contingency_amount: Contingency allocated to this item
        budget_lines: Child budget lines for granularity
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    original_amount: Decimal = Decimal("0.00")
    current_amount: Decimal = Decimal("0.00")
    contingency_amount: Decimal = Decimal("0.00")

    # Linked budget lines for granularity
    budget_lines: List[BudgetLine] = field(default_factory=list)

    def total_budgeted(self) -> Decimal:
        """Sum of current budgets across all linked budget lines."""
        return sum((bl.current_budget for bl in self.budget_lines), Decimal("0.00"))

    def total_actual(self) -> Decimal:
        """Sum of actual costs across all linked budget lines."""
        return sum((bl.actual_cost for bl in self.budget_lines), Decimal("0.00"))

    def total_committed(self) -> Decimal:
        """Sum of committed costs across all linked budget lines."""
        return sum((bl.committed_cost for bl in self.budget_lines), Decimal("0.00"))

    def variance(self) -> Decimal:
        """GMP variance (positive = under, negative = over)."""
        return self.current_amount - self.total_actual()

    def percent_complete(self) -> float:
        """Percentage of GMP spent."""
        if self.current_amount == 0:
            return 0.0
        return float(self.total_actual() / self.current_amount) * 100

    def average_risk_score(self) -> float:
        """Average prospect-weighted risk across budget lines."""
        if not self.budget_lines:
            return 0.0
        return sum(bl.prospect_weighted_risk() for bl in self.budget_lines) / len(self.budget_lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'original_amount': float(self.original_amount),
            'current_amount': float(self.current_amount),
            'contingency_amount': float(self.contingency_amount),
            'total_budgeted': float(self.total_budgeted()),
            'total_actual': float(self.total_actual()),
            'total_committed': float(self.total_committed()),
            'variance': float(self.variance()),
            'percent_complete': self.percent_complete(),
            'average_risk_score': self.average_risk_score(),
            'budget_line_count': len(self.budget_lines),
        }


@dataclass
class GMPAllocation:
    """
    Master GMP allocation for a project.

    Funds flow: Direct Cost -> Budget Line -> GMP Line Item -> GMP Total

    Implements sub-job allocation tracking for temporal cost decomposition.

    Attributes:
        id: Unique identifier
        project_id: Parent project identifier
        project_name: Project name
        total_gmp: Total Guaranteed Maximum Price
        contingency_percent: Contingency percentage (default 5%)
        fee_percent: CM fee percentage (default 3%)
        contract_date: GMP contract execution date
        substantial_completion_date: Target substantial completion
        line_items: GMP line item breakdown
        sub_jobs: Associated sub-jobs
        sub_job_allocations: Mapping of sub-job ID to allocated amount
    """

    id: UUID = field(default_factory=uuid4)
    project_id: UUID = field(default_factory=uuid4)
    project_name: str = ""

    # GMP amounts
    total_gmp: Decimal = Decimal("0.00")
    contingency_percent: Decimal = Decimal("0.05")  # 5% default
    fee_percent: Decimal = Decimal("0.03")  # 3% CM fee

    # Temporal tracking
    contract_date: Optional[date] = None
    substantial_completion_date: Optional[date] = None

    # Breakdown
    line_items: List[GMPLineItem] = field(default_factory=list)

    # Sub-job allocation mapping
    sub_job_allocations: Dict[UUID, Decimal] = field(default_factory=dict)

    def allocate_to_sub_job(self, sub_job_id: UUID, amount: Decimal) -> None:
        """
        Allocate GMP funds to a sub-job.

        Args:
            sub_job_id: Target sub-job UUID
            amount: Amount to allocate (added to existing)
        """
        if sub_job_id in self.sub_job_allocations:
            self.sub_job_allocations[sub_job_id] += amount
        else:
            self.sub_job_allocations[sub_job_id] = amount

    def get_sub_job_allocation(self, sub_job_id: UUID) -> Decimal:
        """Get allocated amount for a sub-job."""
        return self.sub_job_allocations.get(sub_job_id, Decimal("0.00"))

    def total_allocated(self) -> Decimal:
        """Total funds allocated across all sub-jobs."""
        return sum(self.sub_job_allocations.values(), Decimal("0.00"))

    def remaining_unallocated(self) -> Decimal:
        """GMP funds not yet allocated to sub-jobs."""
        return self.total_gmp - self.total_allocated()

    def calculate_contingency(self) -> Decimal:
        """Calculate contingency based on total GMP."""
        return self.total_gmp * self.contingency_percent

    def calculate_fee(self) -> Decimal:
        """Calculate CM fee."""
        return self.total_gmp * self.fee_percent

    def total_budgeted(self) -> Decimal:
        """Sum of all line item budgets."""
        return sum((item.total_budgeted() for item in self.line_items), Decimal("0.00"))

    def total_actual(self) -> Decimal:
        """Sum of all actual costs."""
        return sum((item.total_actual() for item in self.line_items), Decimal("0.00"))

    def total_committed(self) -> Decimal:
        """Sum of all committed costs."""
        return sum((item.total_committed() for item in self.line_items), Decimal("0.00"))

    def variance(self) -> Decimal:
        """Overall GMP variance."""
        return self.total_gmp - self.total_actual()

    def percent_complete(self) -> float:
        """Overall project percent complete by cost."""
        if self.total_gmp == 0:
            return 0.0
        return float(self.total_actual() / self.total_gmp) * 100

    def estimated_at_completion(self) -> Decimal:
        """
        Project-level EAC calculation.

        Uses committed costs plus actuals exceeding commitments.
        """
        eac = Decimal("0.00")
        for item in self.line_items:
            for bl in item.budget_lines:
                eac += bl.estimated_at_completion()
        return eac

    def surplus_or_deficit(self) -> Decimal:
        """
        Calculate projected surplus or deficit.

        Returns:
            Positive = projected surplus
            Negative = projected deficit
        """
        return self.total_gmp - self.estimated_at_completion()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'project_id': str(self.project_id),
            'project_name': self.project_name,
            'total_gmp': float(self.total_gmp),
            'contingency_percent': float(self.contingency_percent),
            'fee_percent': float(self.fee_percent),
            'contract_date': self.contract_date.isoformat() if self.contract_date else None,
            'substantial_completion_date': (
                self.substantial_completion_date.isoformat()
                if self.substantial_completion_date else None
            ),
            'total_budgeted': float(self.total_budgeted()),
            'total_actual': float(self.total_actual()),
            'total_committed': float(self.total_committed()),
            'variance': float(self.variance()),
            'percent_complete': self.percent_complete(),
            'estimated_at_completion': float(self.estimated_at_completion()),
            'surplus_or_deficit': float(self.surplus_or_deficit()),
            'line_item_count': len(self.line_items),
            'sub_job_allocation_count': len(self.sub_job_allocations),
        }

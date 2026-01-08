"""
Budget Line Entity - Intermediary granularity layer.

Purpose: Provide finer granularity than GMP while aggregating direct costs.
Implements prospect theory weighting for risk-adjusted allocations.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional
from uuid import UUID, uuid4
import math

from .direct_cost import DirectCost


@dataclass
class BudgetLine:
    """
    Budget line item aggregating direct costs toward GMP.

    Implements Kahneman-Tversky Prospect Theory for risk-adjusted
    budget variance analysis:
    - Reference point: current_budget
    - Loss aversion (lambda=2.25): Overruns weighted more heavily than savings
    - Diminishing sensitivity (alpha=0.88): Marginal impact decreases

    Attributes:
        id: Unique identifier
        name: Budget line name
        cost_code: CSI code for mapping
        gmp_line_id: Parent GMP line item
        sub_job_id: Associated sub-job
        original_budget: Initial budgeted amount
        current_budget: Current budgeted amount (after changes)
        committed_cost: Contracted/committed costs
        actual_cost: Actual incurred costs
        risk_weight_alpha: Prospect theory curvature parameter (default 0.88)
        loss_aversion_lambda: Loss aversion coefficient (default 2.25)
        direct_costs: Linked direct cost entries
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    cost_code: str = ""  # CSI code for mapping
    gmp_line_id: Optional[UUID] = None
    sub_job_id: Optional[UUID] = None

    # Budget amounts
    original_budget: Decimal = Decimal("0.00")
    current_budget: Decimal = Decimal("0.00")
    committed_cost: Decimal = Decimal("0.00")
    actual_cost: Decimal = Decimal("0.00")

    # Risk parameters (Prospect Theory - Kahneman & Tversky)
    risk_weight_alpha: float = 0.88  # Curvature parameter
    loss_aversion_lambda: float = 2.25  # Loss aversion coefficient

    # Linked direct costs
    direct_costs: List[DirectCost] = field(default_factory=list)

    def aggregate_direct_costs(self) -> Decimal:
        """Sum all linked direct costs."""
        return sum((dc.amount for dc in self.direct_costs), Decimal("0.00"))

    def variance(self) -> Decimal:
        """
        Calculate budget variance.

        Returns:
            Positive = under budget (savings)
            Negative = over budget (overrun)
        """
        return self.current_budget - self.actual_cost

    def variance_percent(self) -> float:
        """Calculate variance as percentage of budget."""
        if self.current_budget == 0:
            return 0.0
        return float(self.variance() / self.current_budget) * 100

    def prospect_weighted_risk(self) -> float:
        """
        Calculate prospect-theory weighted risk score.

        Uses Kahneman-Tversky probability weighting function:
        w(p) = p^alpha / (p^alpha + (1-p)^alpha)^(1/alpha)

        And value function with loss aversion:
        v(x) = x^alpha if x >= 0 (gains)
        v(x) = -lambda * (-x)^alpha if x < 0 (losses)

        Returns:
            Prospect-weighted risk score.
            Positive = favorable (under budget)
            Negative = unfavorable (over budget, amplified by loss aversion)
        """
        if self.current_budget == 0:
            return 0.0

        # Calculate overrun/savings ratio
        variance_ratio = float(self.variance() / self.current_budget)

        # Apply prospect theory value function
        if variance_ratio >= 0:
            # Gain domain (under budget) - diminishing sensitivity
            value = math.pow(abs(variance_ratio), self.risk_weight_alpha)
        else:
            # Loss domain (over budget) - apply loss aversion
            value = -self.loss_aversion_lambda * math.pow(
                abs(variance_ratio), self.risk_weight_alpha
            )

        return value

    def risk_category(self) -> str:
        """
        Categorize risk level based on prospect-weighted score.

        Returns:
            Risk category: 'low', 'medium', 'high', 'critical'
        """
        score = self.prospect_weighted_risk()

        if score >= 0:
            return 'low'  # Under budget
        elif score > -0.5:
            return 'medium'  # Minor overrun
        elif score > -1.0:
            return 'high'  # Significant overrun
        else:
            return 'critical'  # Major overrun (amplified by loss aversion)

    def link_direct_cost(self, cost: DirectCost) -> None:
        """
        Link a direct cost to this budget line.

        Args:
            cost: DirectCost to link

        Side effects:
            - Appends cost to direct_costs list
            - Updates actual_cost
        """
        self.direct_costs.append(cost)
        self.actual_cost += cost.amount

    def estimated_at_completion(self) -> Decimal:
        """
        Calculate estimated cost at completion (EAC).

        Uses current actual cost plus remaining committed costs.
        """
        remaining_committed = max(Decimal("0"), self.committed_cost - self.actual_cost)
        return self.actual_cost + remaining_committed

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'name': self.name,
            'cost_code': self.cost_code,
            'gmp_line_id': str(self.gmp_line_id) if self.gmp_line_id else None,
            'sub_job_id': str(self.sub_job_id) if self.sub_job_id else None,
            'original_budget': float(self.original_budget),
            'current_budget': float(self.current_budget),
            'committed_cost': float(self.committed_cost),
            'actual_cost': float(self.actual_cost),
            'variance': float(self.variance()),
            'variance_percent': self.variance_percent(),
            'prospect_risk': self.prospect_weighted_risk(),
            'risk_category': self.risk_category(),
        }

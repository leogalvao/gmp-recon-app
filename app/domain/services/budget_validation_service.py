"""
Budget Validation Service - Enforces business rules for budget operations.

Implements validation rules from the spec:
- GMP ceiling constraint
- Budget not below actual
- Orphan cost prevention
"""
from typing import Optional, List, Dict, Tuple
from sqlalchemy.orm import Session

from app.models import GMP, BudgetEntity, DirectCostEntity, ChangeOrder, ChangeOrderStatus
from app.infrastructure.repositories import (
    GMPRepository,
    BudgetRepository,
    DirectCostRepository,
)
from app.domain.exceptions import (
    GMPCeilingExceededError,
    BudgetUnderflowError,
    BudgetHasMappedCostsError,
    GMPNotFoundError,
    BudgetNotFoundError,
    ValidationError,
)


class BudgetValidationService:
    """
    Service for validating budget operations.

    Ensures all budget modifications comply with business rules:
    - GMP ceiling is not exceeded
    - Budget cannot be reduced below actual spent
    - Budget cannot be deleted if it has mapped costs
    """

    def __init__(self, session: Session):
        self.session = session
        self.gmp_repo = GMPRepository(session)
        self.budget_repo = BudgetRepository(session)
        self.cost_repo = DirectCostRepository(session)

    # =========================================================================
    # GMP Ceiling Validation
    # =========================================================================

    def get_gmp_ceiling(self, gmp_id: int) -> int:
        """
        Get the authorized GMP ceiling amount.

        Ceiling = Original Amount + Approved Change Orders

        Args:
            gmp_id: GMP identifier

        Returns:
            Authorized ceiling in cents

        Raises:
            GMPNotFoundError: If GMP not found
        """
        gmp = self.session.query(GMP).filter(GMP.id == gmp_id).first()
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))

        return gmp.authorized_amount_cents

    def get_available_budget(
        self,
        gmp_id: int,
        exclude_budget_id: Optional[int] = None
    ) -> int:
        """
        Get available budget room under GMP ceiling.

        Args:
            gmp_id: GMP identifier
            exclude_budget_id: Budget to exclude from sum (for updates)

        Returns:
            Available amount in cents
        """
        ceiling = self.get_gmp_ceiling(gmp_id)
        current_total = self._get_total_budgeted(gmp_id, exclude_budget_id)
        return ceiling - current_total

    def _get_total_budgeted(
        self,
        gmp_id: int,
        exclude_budget_id: Optional[int] = None
    ) -> int:
        """Get sum of all budgets for a GMP."""
        query = self.session.query(BudgetEntity).filter(
            BudgetEntity.gmp_id == gmp_id
        )
        if exclude_budget_id:
            query = query.filter(BudgetEntity.id != exclude_budget_id)

        return sum(b.current_budget_cents or 0 for b in query.all())

    def validate_gmp_ceiling(
        self,
        gmp_id: int,
        new_budget_amount: int,
        exclude_budget_id: Optional[int] = None
    ) -> bool:
        """
        Validate that a budget amount won't exceed GMP ceiling.

        Implements: Σ(Budget.current_budget | gmp_id = G) ≤ GMP[G].authorized_amount

        Args:
            gmp_id: GMP identifier
            new_budget_amount: Amount to validate (in cents)
            exclude_budget_id: Budget ID to exclude (for updates)

        Returns:
            True if valid

        Raises:
            GMPCeilingExceededError: If ceiling would be exceeded
        """
        ceiling = self.get_gmp_ceiling(gmp_id)
        current_total = self._get_total_budgeted(gmp_id, exclude_budget_id)
        new_total = current_total + new_budget_amount

        if new_total > ceiling:
            raise GMPCeilingExceededError(
                total_budgeted=new_total,
                gmp_amount=ceiling,
                available=ceiling - current_total
            )

        return True

    # =========================================================================
    # Budget Underflow Validation
    # =========================================================================

    def get_actual_cost(self, budget_id: int) -> int:
        """
        Get actual cost for a budget.

        Args:
            budget_id: Budget identifier

        Returns:
            Total actual cost in cents
        """
        return self.cost_repo.get_total_by_budget(budget_id)

    def validate_not_below_actual(
        self,
        budget_id: int,
        new_amount: int
    ) -> bool:
        """
        Validate that budget is not reduced below actual spent.

        Implements: current_budget_cents >= actual_cost_cents

        Args:
            budget_id: Budget identifier
            new_amount: Proposed new budget amount

        Returns:
            True if valid

        Raises:
            BudgetUnderflowError: If amount is below actual
        """
        actual = self.get_actual_cost(budget_id)

        if new_amount < actual:
            raise BudgetUnderflowError(
                budget_amount=new_amount,
                actual_cost=actual
            )

        return True

    # =========================================================================
    # Orphan Cost Prevention
    # =========================================================================

    def get_mapped_cost_count(self, budget_id: int) -> int:
        """
        Get count of direct costs mapped to a budget.

        Args:
            budget_id: Budget identifier

        Returns:
            Count of mapped costs
        """
        return self.session.query(DirectCostEntity).filter(
            DirectCostEntity.mapped_budget_id == budget_id
        ).count()

    def validate_no_orphan_costs(self, budget_id: int) -> bool:
        """
        Validate that deleting a budget won't orphan costs.

        Args:
            budget_id: Budget identifier

        Returns:
            True if safe to delete

        Raises:
            BudgetHasMappedCostsError: If budget has mapped costs
        """
        cost_count = self.get_mapped_cost_count(budget_id)

        if cost_count > 0:
            raise BudgetHasMappedCostsError(str(budget_id), cost_count)

        return True

    # =========================================================================
    # Comprehensive Validation
    # =========================================================================

    def validate_budget_create(
        self,
        gmp_id: int,
        amount_cents: int,
        cost_code: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate all rules for creating a budget.

        Args:
            gmp_id: Parent GMP identifier
            amount_cents: Budget amount
            cost_code: CSI cost code

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Validate GMP exists
        try:
            self.get_gmp_ceiling(gmp_id)
        except GMPNotFoundError:
            errors.append(f"GMP {gmp_id} not found")
            return False, errors

        # Validate positive amount
        if amount_cents <= 0:
            errors.append("Budget amount must be positive")

        # Validate ceiling constraint
        try:
            self.validate_gmp_ceiling(gmp_id, amount_cents)
        except GMPCeilingExceededError as e:
            errors.append(e.message)

        # Validate cost code format (basic check)
        if not cost_code or len(cost_code.strip()) == 0:
            errors.append("Cost code is required")

        return len(errors) == 0, errors

    def validate_budget_update(
        self,
        budget_id: int,
        new_amount_cents: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate all rules for updating a budget.

        Args:
            budget_id: Budget identifier
            new_amount_cents: New budget amount

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Get budget
        budget = self.budget_repo.get_by_id(budget_id)
        if not budget:
            errors.append(f"Budget {budget_id} not found")
            return False, errors

        # Validate positive amount
        if new_amount_cents < 0:
            errors.append("Budget amount cannot be negative")

        # Validate not below actual
        try:
            self.validate_not_below_actual(budget_id, new_amount_cents)
        except BudgetUnderflowError as e:
            errors.append(e.message)

        # Validate ceiling constraint
        try:
            self.validate_gmp_ceiling(
                budget.gmp_id,
                new_amount_cents,
                exclude_budget_id=budget_id
            )
        except GMPCeilingExceededError as e:
            errors.append(e.message)

        return len(errors) == 0, errors

    def validate_budget_delete(self, budget_id: int) -> Tuple[bool, List[str]]:
        """
        Validate all rules for deleting a budget.

        Args:
            budget_id: Budget identifier

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Get budget
        budget = self.budget_repo.get_by_id(budget_id)
        if not budget:
            errors.append(f"Budget {budget_id} not found")
            return False, errors

        # Validate no orphan costs
        try:
            self.validate_no_orphan_costs(budget_id)
        except BudgetHasMappedCostsError as e:
            errors.append(e.message)

        return len(errors) == 0, errors

    # =========================================================================
    # Budget Transfer Validation
    # =========================================================================

    def validate_budget_transfer(
        self,
        from_budget_id: int,
        to_budget_id: int,
        amount_cents: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate a budget transfer between two budgets.

        Args:
            from_budget_id: Source budget
            to_budget_id: Destination budget
            amount_cents: Amount to transfer

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Get both budgets
        from_budget = self.budget_repo.get_by_id(from_budget_id)
        to_budget = self.budget_repo.get_by_id(to_budget_id)

        if not from_budget:
            errors.append(f"Source budget {from_budget_id} not found")
        if not to_budget:
            errors.append(f"Destination budget {to_budget_id} not found")

        if errors:
            return False, errors

        # Validate same GMP
        if from_budget.gmp_id != to_budget.gmp_id:
            errors.append("Cannot transfer between budgets under different GMPs")
            return False, errors

        # Validate positive amount
        if amount_cents <= 0:
            errors.append("Transfer amount must be positive")

        # Validate source has enough budget
        new_from_amount = from_budget.current_budget_cents - amount_cents
        try:
            self.validate_not_below_actual(from_budget_id, new_from_amount)
        except BudgetUnderflowError as e:
            errors.append(f"Source budget: {e.message}")

        # Note: GMP ceiling doesn't apply since it's a transfer within same GMP

        return len(errors) == 0, errors

    # =========================================================================
    # Summary and Reporting
    # =========================================================================

    def get_validation_summary(self, gmp_id: int) -> Dict:
        """
        Get validation summary for a GMP and its budgets.

        Args:
            gmp_id: GMP identifier

        Returns:
            Dict with ceiling status, budget health, warnings
        """
        try:
            ceiling = self.get_gmp_ceiling(gmp_id)
        except GMPNotFoundError:
            return {'error': f'GMP {gmp_id} not found'}

        total_budgeted = self._get_total_budgeted(gmp_id)
        budgets = self.budget_repo.get_by_gmp(gmp_id)

        budget_health = []
        warnings = []

        for budget in budgets:
            actual = self.get_actual_cost(budget.id)
            remaining = budget.current_budget_cents - actual
            pct_spent = (actual / budget.current_budget_cents * 100) if budget.current_budget_cents > 0 else 0

            status = 'healthy'
            if pct_spent > 100:
                status = 'over_budget'
                warnings.append(f"Budget {budget.cost_code} is over budget")
            elif pct_spent > 90:
                status = 'warning'
                warnings.append(f"Budget {budget.cost_code} is at {pct_spent:.1f}% spent")

            budget_health.append({
                'budget_id': budget.id,
                'cost_code': budget.cost_code,
                'current_budget_cents': budget.current_budget_cents,
                'actual_cost_cents': actual,
                'remaining_cents': remaining,
                'percent_spent': round(pct_spent, 2),
                'status': status,
            })

        ceiling_pct = (total_budgeted / ceiling * 100) if ceiling > 0 else 0

        return {
            'gmp_id': gmp_id,
            'ceiling_cents': ceiling,
            'total_budgeted_cents': total_budgeted,
            'available_cents': ceiling - total_budgeted,
            'ceiling_utilization_pct': round(ceiling_pct, 2),
            'budget_count': len(budgets),
            'budgets': budget_health,
            'warnings': warnings,
            'is_ceiling_exceeded': total_budgeted > ceiling,
        }

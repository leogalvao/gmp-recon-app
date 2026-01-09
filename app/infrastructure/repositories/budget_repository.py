"""
Budget Repository - Data access layer for Budget entities.

Implements repository pattern for Budget operations with:
- GMP ceiling validation
- Actual cost aggregation
- Mapping tracking
"""
import uuid
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import BudgetEntity, DirectCostEntity, GMP
from app.domain.exceptions import (
    BudgetNotFoundError,
    BudgetHasMappedCostsError,
    GMPCeilingExceededError,
    BudgetUnderflowError,
    GMPNotFoundError,
)
from .base_repository import BaseRepository


class BudgetRepository(BaseRepository[BudgetEntity]):
    """
    Repository for Budget entities.

    Budget lines are fully editable but must respect:
    - GMP ceiling constraint (sum of budgets <= GMP authorized amount)
    - Cannot reduce below actual spent
    """

    def __init__(self, session: Session):
        super().__init__(session, BudgetEntity)

    def exists(self, **criteria) -> bool:
        """Check if a Budget matching the criteria exists."""
        query = self.session.query(BudgetEntity)
        for field, value in criteria.items():
            query = query.filter(getattr(BudgetEntity, field) == value)
        return query.first() is not None

    def get_by_gmp(self, gmp_id: int) -> List[BudgetEntity]:
        """
        Get all budgets for a GMP.

        Args:
            gmp_id: Parent GMP identifier

        Returns:
            List of budget entities
        """
        return self.session.query(BudgetEntity).filter(
            BudgetEntity.gmp_id == gmp_id
        ).order_by(BudgetEntity.cost_code).all()

    def get_by_cost_code(self, cost_code: str) -> List[BudgetEntity]:
        """
        Get all budgets with a specific cost code.

        Args:
            cost_code: CSI cost code

        Returns:
            List of budget entities
        """
        return self.session.query(BudgetEntity).filter(
            BudgetEntity.cost_code == cost_code
        ).all()

    def get_by_zone(self, zone: str) -> List[BudgetEntity]:
        """
        Get all budgets for a zone.

        Args:
            zone: Zone identifier (EAST, WEST, SHARED)

        Returns:
            List of budget entities
        """
        return self.session.query(BudgetEntity).filter(
            BudgetEntity.zone == zone
        ).all()

    def _get_gmp_ceiling(self, gmp_id: int) -> int:
        """Get authorized GMP ceiling amount."""
        gmp = self.session.query(GMP).filter(GMP.id == gmp_id).first()
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))
        return gmp.authorized_amount_cents

    def _get_sibling_budgets_total(
        self,
        gmp_id: int,
        exclude_budget_id: Optional[int] = None
    ) -> int:
        """Get sum of sibling budgets under the same GMP."""
        query = self.session.query(
            func.coalesce(func.sum(BudgetEntity.current_budget_cents), 0)
        ).filter(BudgetEntity.gmp_id == gmp_id)

        if exclude_budget_id:
            query = query.filter(BudgetEntity.id != exclude_budget_id)

        return query.scalar() or 0

    def validate_gmp_ceiling(
        self,
        gmp_id: int,
        new_amount: int,
        exclude_budget_id: Optional[int] = None
    ) -> bool:
        """
        Validate that new budget amount won't exceed GMP ceiling.

        Args:
            gmp_id: Parent GMP identifier
            new_amount: New budget amount in cents
            exclude_budget_id: Budget ID to exclude from sum (for updates)

        Returns:
            True if valid

        Raises:
            GMPCeilingExceededError if ceiling would be exceeded
        """
        ceiling = self._get_gmp_ceiling(gmp_id)
        existing_total = self._get_sibling_budgets_total(gmp_id, exclude_budget_id)
        new_total = existing_total + new_amount

        if new_total > ceiling:
            raise GMPCeilingExceededError(
                total_budgeted=new_total,
                gmp_amount=ceiling,
                available=ceiling - existing_total
            )
        return True

    def get_actual_cost(self, budget_id: int) -> int:
        """
        Get sum of all direct costs mapped to this budget.

        Args:
            budget_id: Budget identifier

        Returns:
            Total actual cost in cents
        """
        result = self.session.query(
            func.coalesce(func.sum(DirectCostEntity.gross_amount_cents), 0)
        ).filter(
            DirectCostEntity.mapped_budget_id == budget_id
        ).scalar()
        return result or 0

    def get_mapped_cost_count(self, budget_id: int) -> int:
        """
        Get count of direct costs mapped to this budget.

        Args:
            budget_id: Budget identifier

        Returns:
            Count of mapped costs
        """
        return self.session.query(DirectCostEntity).filter(
            DirectCostEntity.mapped_budget_id == budget_id
        ).count()

    def create(
        self,
        gmp_id: int,
        cost_code: str,
        current_budget_cents: int,
        zone: Optional[str] = None,
        description: Optional[str] = None,
        committed_cents: int = 0
    ) -> BudgetEntity:
        """
        Create a new budget entity.

        Args:
            gmp_id: Parent GMP identifier
            cost_code: CSI cost code
            current_budget_cents: Budget amount in cents
            zone: Optional zone assignment
            description: Optional description
            committed_cents: Committed amount in cents

        Returns:
            Created budget entity

        Raises:
            GMPNotFoundError: If GMP not found
            GMPCeilingExceededError: If ceiling would be exceeded
        """
        # Validate GMP exists
        gmp = self.session.query(GMP).filter(GMP.id == gmp_id).first()
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))

        # Validate ceiling constraint
        self.validate_gmp_ceiling(gmp_id, current_budget_cents)

        budget = BudgetEntity(
            uuid=str(uuid.uuid4()),
            gmp_id=gmp_id,
            cost_code=cost_code,
            current_budget_cents=current_budget_cents,
            zone=zone,
            description=description,
            committed_cents=committed_cents,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.add(budget)
        return budget

    def update_amount(
        self,
        budget_id: int,
        new_amount_cents: int
    ) -> BudgetEntity:
        """
        Update budget amount.

        Args:
            budget_id: Budget identifier
            new_amount_cents: New budget amount in cents

        Returns:
            Updated budget entity

        Raises:
            BudgetNotFoundError: If budget not found
            GMPCeilingExceededError: If ceiling would be exceeded
            BudgetUnderflowError: If amount is below actual spent
        """
        budget = self.get_by_id(budget_id)
        if not budget:
            raise BudgetNotFoundError(str(budget_id))

        # Validate not below actual
        actual_cost = self.get_actual_cost(budget_id)
        if new_amount_cents < actual_cost:
            raise BudgetUnderflowError(new_amount_cents, actual_cost)

        # Validate ceiling constraint
        self.validate_gmp_ceiling(
            budget.gmp_id,
            new_amount_cents,
            exclude_budget_id=budget_id
        )

        budget.current_budget_cents = new_amount_cents
        budget.updated_at = datetime.utcnow()
        return budget

    def update_zone(self, budget_id: int, zone: str) -> BudgetEntity:
        """
        Update budget zone assignment.

        Args:
            budget_id: Budget identifier
            zone: New zone (EAST, WEST, SHARED)

        Returns:
            Updated budget entity

        Raises:
            BudgetNotFoundError: If budget not found
        """
        budget = self.get_by_id(budget_id)
        if not budget:
            raise BudgetNotFoundError(str(budget_id))

        budget.zone = zone
        budget.updated_at = datetime.utcnow()
        return budget

    def delete_budget(self, budget_id: int) -> None:
        """
        Delete a budget entity.

        Args:
            budget_id: Budget identifier

        Raises:
            BudgetNotFoundError: If budget not found
            BudgetHasMappedCostsError: If budget has mapped costs
        """
        budget = self.get_by_id(budget_id)
        if not budget:
            raise BudgetNotFoundError(str(budget_id))

        # Check for mapped costs
        cost_count = self.get_mapped_cost_count(budget_id)
        if cost_count > 0:
            raise BudgetHasMappedCostsError(str(budget_id), cost_count)

        self.delete(budget)

    def get_summary(self, budget_id: int) -> Dict:
        """
        Get comprehensive budget summary with computed fields.

        Args:
            budget_id: Budget identifier

        Returns:
            Dictionary with budget data and computed metrics
        """
        budget = self.get_by_id(budget_id)
        if not budget:
            raise BudgetNotFoundError(str(budget_id))

        actual_cost = self.get_actual_cost(budget_id)
        remaining = budget.current_budget_cents - actual_cost

        return {
            'id': budget.id,
            'uuid': budget.uuid,
            'gmp_id': budget.gmp_id,
            'cost_code': budget.cost_code,
            'zone': budget.zone,
            'description': budget.description,
            'current_budget_cents': budget.current_budget_cents,
            'committed_cents': budget.committed_cents,
            'actual_cost_cents': actual_cost,
            'remaining_cents': remaining,
            'percent_spent': (actual_cost / budget.current_budget_cents * 100) if budget.current_budget_cents > 0 else 0,
            'mapped_cost_count': self.get_mapped_cost_count(budget_id),
            'created_at': budget.created_at.isoformat() if budget.created_at else None,
            'updated_at': budget.updated_at.isoformat() if budget.updated_at else None,
        }

    def get_all_for_gmp_with_actuals(self, gmp_id: int) -> List[Dict]:
        """
        Get all budgets for a GMP with actual cost aggregations.

        Args:
            gmp_id: GMP identifier

        Returns:
            List of budget summaries
        """
        budgets = self.get_by_gmp(gmp_id)
        return [self.get_summary(budget.id) for budget in budgets]

    def get_unmapped_to_zone(self) -> List[BudgetEntity]:
        """
        Get all budgets without zone assignment.

        Returns:
            List of budgets with null zone
        """
        return self.session.query(BudgetEntity).filter(
            BudgetEntity.zone.is_(None)
        ).all()

    def bulk_update_zone(self, budget_ids: List[int], zone: str) -> int:
        """
        Bulk update zone for multiple budgets.

        Args:
            budget_ids: List of budget identifiers
            zone: Zone to assign

        Returns:
            Count of updated budgets
        """
        count = self.session.query(BudgetEntity).filter(
            BudgetEntity.id.in_(budget_ids)
        ).update({'zone': zone, 'updated_at': datetime.utcnow()}, synchronize_session=False)
        return count

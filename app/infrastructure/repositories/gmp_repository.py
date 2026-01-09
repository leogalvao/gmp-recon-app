"""
GMP Repository - Data access layer for GMP entities.

Implements repository pattern for GMP operations with:
- Immutability enforcement
- Change order integration
- Computed field helpers
"""
import uuid
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import GMP, BudgetEntity, DirectCostEntity, ChangeOrder, ChangeOrderStatus
from app.domain.exceptions import (
    GMPNotFoundError,
    DuplicateGMPError,
    ImmutableFieldError,
)
from .base_repository import BaseRepository


class GMPRepository(BaseRepository[GMP]):
    """
    Repository for GMP (Guaranteed Maximum Price) entities.

    GMP is immutable after creation - only description can be updated.
    Amount changes must go through Change Orders.
    """

    def __init__(self, session: Session):
        super().__init__(session, GMP)

    def exists(self, **criteria) -> bool:
        """Check if a GMP matching the criteria exists."""
        query = self.session.query(GMP)
        for field, value in criteria.items():
            query = query.filter(getattr(GMP, field) == value)
        return query.first() is not None

    def get_by_division_and_zone(
        self,
        project_id: int,
        division: str,
        zone: str
    ) -> Optional[GMP]:
        """
        Get GMP by project, division, and zone.

        Args:
            project_id: Project identifier
            division: CSI division name
            zone: Zone identifier (EAST, WEST, SHARED)

        Returns:
            GMP entity if found, None otherwise
        """
        return self.session.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.division == division,
            GMP.zone == zone
        ).first()

    def get_by_project(self, project_id: int) -> List[GMP]:
        """
        Get all GMPs for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of GMP entities
        """
        return self.session.query(GMP).filter(
            GMP.project_id == project_id
        ).order_by(GMP.division).all()

    def get_by_division(self, division: str) -> List[GMP]:
        """
        Get all GMPs for a division (across all zones).

        Args:
            division: CSI division name

        Returns:
            List of GMP entities
        """
        return self.session.query(GMP).filter(
            GMP.division == division
        ).all()

    def create(
        self,
        project_id: int,
        division: str,
        zone: str,
        original_amount_cents: int,
        description: Optional[str] = None
    ) -> GMP:
        """
        Create a new GMP entity.

        Args:
            project_id: Parent project ID
            division: CSI division name
            zone: Zone identifier
            original_amount_cents: Immutable GMP amount in cents
            description: Optional description

        Returns:
            Created GMP entity

        Raises:
            DuplicateGMPError: If GMP already exists for division/zone
            ValueError: If amount is not positive
        """
        # Validate uniqueness
        if self.exists(project_id=project_id, division=division, zone=zone):
            raise DuplicateGMPError(division, str(project_id))

        # Validate positive amount
        if original_amount_cents <= 0:
            raise ValueError("GMP amount must be positive")

        gmp = GMP(
            uuid=str(uuid.uuid4()),
            project_id=project_id,
            division=division,
            zone=zone,
            original_amount_cents=original_amount_cents,
            description=description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.add(gmp)
        return gmp

    def update_description(self, gmp_id: int, description: str) -> GMP:
        """
        Update GMP description (the only mutable field).

        Args:
            gmp_id: GMP identifier
            description: New description

        Returns:
            Updated GMP entity

        Raises:
            GMPNotFoundError: If GMP not found
        """
        gmp = self.get_by_id(gmp_id)
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))

        gmp.description = description
        gmp.updated_at = datetime.utcnow()
        return gmp

    # Computed field helpers

    def get_total_budgeted(self, gmp_id: int) -> int:
        """
        Get sum of all budgets for a GMP.

        Args:
            gmp_id: GMP identifier

        Returns:
            Total budgeted amount in cents
        """
        result = self.session.query(
            func.coalesce(func.sum(BudgetEntity.current_budget_cents), 0)
        ).filter(BudgetEntity.gmp_id == gmp_id).scalar()
        return result or 0

    def get_total_actual(self, gmp_id: int) -> int:
        """
        Get sum of all actual costs for a GMP.

        Aggregates through Budget → DirectCost relationship.

        Args:
            gmp_id: GMP identifier

        Returns:
            Total actual cost in cents
        """
        result = self.session.query(
            func.coalesce(func.sum(DirectCostEntity.gross_amount_cents), 0)
        ).join(
            BudgetEntity, DirectCostEntity.mapped_budget_id == BudgetEntity.id
        ).filter(
            BudgetEntity.gmp_id == gmp_id
        ).scalar()
        return result or 0

    def get_change_order_total(self, gmp_id: int, status: Optional[str] = None) -> int:
        """
        Get sum of change orders for a GMP.

        Args:
            gmp_id: GMP identifier
            status: Optional filter by status (default: approved only)

        Returns:
            Total change order amount in cents
        """
        query = self.session.query(
            func.coalesce(func.sum(ChangeOrder.amount_cents), 0)
        ).filter(ChangeOrder.gmp_id == gmp_id)

        if status:
            query = query.filter(ChangeOrder.status == status)
        else:
            query = query.filter(ChangeOrder.status == ChangeOrderStatus.APPROVED.value)

        return query.scalar() or 0

    def get_authorized_amount(self, gmp_id: int) -> int:
        """
        Get authorized GMP amount (original + approved COs).

        Args:
            gmp_id: GMP identifier

        Returns:
            Authorized amount in cents
        """
        gmp = self.get_by_id(gmp_id)
        if not gmp:
            return 0
        return gmp.original_amount_cents + self.get_change_order_total(gmp_id)

    def get_remaining(self, gmp_id: int) -> int:
        """
        Get remaining GMP amount.

        Args:
            gmp_id: GMP identifier

        Returns:
            Remaining amount in cents (authorized - actual)
        """
        return self.get_authorized_amount(gmp_id) - self.get_total_actual(gmp_id)

    def get_budget_utilization_pct(self, gmp_id: int) -> float:
        """
        Get budget utilization percentage.

        Args:
            gmp_id: GMP identifier

        Returns:
            Percentage of GMP allocated to budgets
        """
        authorized = self.get_authorized_amount(gmp_id)
        if authorized == 0:
            return 0.0
        return (self.get_total_budgeted(gmp_id) / authorized) * 100

    def get_summary(self, gmp_id: int) -> Dict:
        """
        Get comprehensive GMP summary with all computed fields.

        Args:
            gmp_id: GMP identifier

        Returns:
            Dictionary with GMP data and computed metrics
        """
        gmp = self.get_by_id(gmp_id)
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))

        authorized = self.get_authorized_amount(gmp_id)
        total_budgeted = self.get_total_budgeted(gmp_id)
        total_actual = self.get_total_actual(gmp_id)
        co_total = self.get_change_order_total(gmp_id)

        return {
            'id': gmp.id,
            'uuid': gmp.uuid,
            'division': gmp.division,
            'zone': gmp.zone,
            'description': gmp.description,
            'original_amount_cents': gmp.original_amount_cents,
            'change_order_total_cents': co_total,
            'authorized_amount_cents': authorized,
            'total_budgeted_cents': total_budgeted,
            'total_actual_cents': total_actual,
            'remaining_cents': authorized - total_actual,
            'budget_utilization_pct': (total_budgeted / authorized * 100) if authorized > 0 else 0,
            'spent_pct': (total_actual / authorized * 100) if authorized > 0 else 0,
            'created_at': gmp.created_at.isoformat() if gmp.created_at else None,
            'updated_at': gmp.updated_at.isoformat() if gmp.updated_at else None,
        }

    def get_all_with_totals(self, project_id: int) -> List[Dict]:
        """
        Get all GMPs for a project with computed totals.

        Args:
            project_id: Project identifier

        Returns:
            List of GMP summaries
        """
        gmps = self.get_by_project(project_id)
        return [self.get_summary(gmp.id) for gmp in gmps]

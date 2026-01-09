"""
Direct Cost Repository - Data access layer for Direct Cost entities.

Implements repository pattern for Direct Cost operations with:
- Budget mapping
- Period aggregation
- Cascade update tracking
"""
import uuid
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date
from collections import defaultdict
from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from app.models import DirectCostEntity, BudgetEntity, GMP
from app.domain.exceptions import (
    DirectCostNotFoundError,
    InvalidMappingError,
    BudgetNotFoundError,
)
from .base_repository import BaseRepository


class DirectCostRepository(BaseRepository[DirectCostEntity]):
    """
    Repository for Direct Cost entities.

    Direct costs are fully editable. Changes trigger recalculation
    of parent Budget and GMP aggregates.
    """

    def __init__(self, session: Session):
        super().__init__(session, DirectCostEntity)

    def exists(self, **criteria) -> bool:
        """Check if a DirectCost matching the criteria exists."""
        query = self.session.query(DirectCostEntity)
        for field, value in criteria.items():
            query = query.filter(getattr(DirectCostEntity, field) == value)
        return query.first() is not None

    def get_by_budget(self, budget_id: int) -> List[DirectCostEntity]:
        """
        Get all direct costs mapped to a budget.

        Args:
            budget_id: Budget identifier

        Returns:
            List of direct cost entities
        """
        return self.session.query(DirectCostEntity).filter(
            DirectCostEntity.mapped_budget_id == budget_id
        ).order_by(DirectCostEntity.transaction_date.desc()).all()

    def get_unmapped(self) -> List[DirectCostEntity]:
        """
        Get all unmapped direct costs.

        Returns:
            List of direct costs without budget mapping
        """
        return self.session.query(DirectCostEntity).filter(
            DirectCostEntity.mapped_budget_id.is_(None)
        ).order_by(DirectCostEntity.transaction_date.desc()).all()

    def get_by_vendor(self, vendor_normalized: str) -> List[DirectCostEntity]:
        """
        Get all direct costs for a vendor.

        Args:
            vendor_normalized: Normalized vendor name

        Returns:
            List of direct cost entities
        """
        return self.session.query(DirectCostEntity).filter(
            DirectCostEntity.vendor_normalized == vendor_normalized
        ).all()

    def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        budget_id: Optional[int] = None
    ) -> List[DirectCostEntity]:
        """
        Get direct costs within a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            budget_id: Optional budget filter

        Returns:
            List of direct cost entities
        """
        query = self.session.query(DirectCostEntity).filter(
            DirectCostEntity.transaction_date >= start_date,
            DirectCostEntity.transaction_date <= end_date
        )

        if budget_id:
            query = query.filter(DirectCostEntity.mapped_budget_id == budget_id)

        return query.order_by(DirectCostEntity.transaction_date).all()

    def create(
        self,
        gross_amount_cents: int,
        vendor_name: Optional[str] = None,
        description: Optional[str] = None,
        transaction_date: Optional[date] = None,
        mapped_budget_id: Optional[int] = None,
        vendor_normalized: Optional[str] = None,
        retainage_amount_cents: int = 0,
        allocation_method: str = 'direct',
        source_row_id: Optional[int] = None
    ) -> DirectCostEntity:
        """
        Create a new direct cost entity.

        Args:
            gross_amount_cents: Cost amount in cents
            vendor_name: Vendor name
            description: Transaction description
            transaction_date: Date of transaction
            mapped_budget_id: Budget to map to (optional)
            vendor_normalized: Normalized vendor name for matching
            retainage_amount_cents: Retainage amount in cents
            allocation_method: Allocation method (direct, split_50_50)
            source_row_id: Source file row ID

        Returns:
            Created direct cost entity
        """
        # Validate budget exists if provided
        zone = None
        if mapped_budget_id:
            budget = self.session.query(BudgetEntity).filter(
                BudgetEntity.id == mapped_budget_id
            ).first()
            if not budget:
                raise BudgetNotFoundError(str(mapped_budget_id))
            zone = budget.zone

        cost = DirectCostEntity(
            uuid=str(uuid.uuid4()),
            gross_amount_cents=gross_amount_cents,
            vendor_name=vendor_name,
            vendor_normalized=vendor_normalized or (
                vendor_name.lower().strip() if vendor_name else None
            ),
            description=description,
            transaction_date=transaction_date or date.today(),
            mapped_budget_id=mapped_budget_id,
            retainage_amount_cents=retainage_amount_cents,
            allocation_method=allocation_method,
            zone=zone,
            source_row_id=source_row_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.add(cost)
        return cost

    def update_mapping(
        self,
        cost_id: int,
        new_budget_id: Optional[int]
    ) -> Tuple[DirectCostEntity, Optional[int]]:
        """
        Update direct cost budget mapping.

        Args:
            cost_id: Direct cost identifier
            new_budget_id: New budget ID (None to unmap)

        Returns:
            Tuple of (updated cost, old budget ID for cascade)

        Raises:
            DirectCostNotFoundError: If cost not found
            BudgetNotFoundError: If new budget not found
        """
        cost = self.get_by_id(cost_id)
        if not cost:
            raise DirectCostNotFoundError(str(cost_id))

        old_budget_id = cost.mapped_budget_id

        # Validate new budget exists
        zone = None
        if new_budget_id:
            budget = self.session.query(BudgetEntity).filter(
                BudgetEntity.id == new_budget_id
            ).first()
            if not budget:
                raise BudgetNotFoundError(str(new_budget_id))
            zone = budget.zone

        cost.mapped_budget_id = new_budget_id
        cost.zone = zone
        cost.updated_at = datetime.utcnow()

        return cost, old_budget_id

    def update_amount(self, cost_id: int, new_amount_cents: int) -> DirectCostEntity:
        """
        Update direct cost amount.

        Args:
            cost_id: Direct cost identifier
            new_amount_cents: New amount in cents

        Returns:
            Updated cost entity

        Raises:
            DirectCostNotFoundError: If cost not found
        """
        cost = self.get_by_id(cost_id)
        if not cost:
            raise DirectCostNotFoundError(str(cost_id))

        cost.gross_amount_cents = new_amount_cents
        cost.updated_at = datetime.utcnow()
        return cost

    def bulk_map(
        self,
        mappings: List[Dict[str, int]]
    ) -> Tuple[int, List[int]]:
        """
        Bulk map direct costs to budgets.

        Args:
            mappings: List of {cost_id, budget_id} dicts

        Returns:
            Tuple of (count updated, list of affected budget IDs)
        """
        affected_budgets = set()
        updated = 0

        for mapping in mappings:
            cost_id = mapping.get('direct_cost_id') or mapping.get('cost_id')
            budget_id = mapping.get('budget_id')

            cost = self.get_by_id(cost_id)
            if cost:
                if cost.mapped_budget_id:
                    affected_budgets.add(cost.mapped_budget_id)
                if budget_id:
                    affected_budgets.add(budget_id)

                cost.mapped_budget_id = budget_id
                cost.updated_at = datetime.utcnow()

                # Update zone from budget
                if budget_id:
                    budget = self.session.query(BudgetEntity).filter(
                        BudgetEntity.id == budget_id
                    ).first()
                    if budget:
                        cost.zone = budget.zone
                else:
                    cost.zone = None

                updated += 1

        return updated, list(affected_budgets)

    # Aggregation methods

    def get_total_by_budget(self, budget_id: int) -> int:
        """Get sum of costs for a budget."""
        result = self.session.query(
            func.coalesce(func.sum(DirectCostEntity.gross_amount_cents), 0)
        ).filter(
            DirectCostEntity.mapped_budget_id == budget_id
        ).scalar()
        return result or 0

    def get_total_unmapped(self) -> int:
        """Get sum of unmapped costs."""
        result = self.session.query(
            func.coalesce(func.sum(DirectCostEntity.gross_amount_cents), 0)
        ).filter(
            DirectCostEntity.mapped_budget_id.is_(None)
        ).scalar()
        return result or 0

    def get_period_totals(
        self,
        budget_id: Optional[int] = None,
        granularity: str = 'month'
    ) -> Dict[str, int]:
        """
        Get cost totals by period.

        Args:
            budget_id: Optional budget filter
            granularity: 'week' or 'month'

        Returns:
            Dict mapping period key to total cents
        """
        query = self.session.query(DirectCostEntity)

        if budget_id:
            query = query.filter(DirectCostEntity.mapped_budget_id == budget_id)

        costs = query.filter(
            DirectCostEntity.transaction_date.isnot(None)
        ).all()

        totals = defaultdict(int)
        for cost in costs:
            if granularity == 'week':
                iso_cal = cost.transaction_date.isocalendar()
                period_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
            else:  # month
                period_key = cost.transaction_date.strftime('%Y-%m')

            totals[period_key] += cost.gross_amount_cents

        return dict(totals)

    def get_cumulative_by_date(
        self,
        as_of_date: date,
        budget_id: Optional[int] = None
    ) -> int:
        """
        Get cumulative cost up to a date.

        Args:
            as_of_date: Date to calculate through
            budget_id: Optional budget filter

        Returns:
            Cumulative cost in cents
        """
        query = self.session.query(
            func.coalesce(func.sum(DirectCostEntity.gross_amount_cents), 0)
        ).filter(
            DirectCostEntity.transaction_date <= as_of_date
        )

        if budget_id:
            query = query.filter(DirectCostEntity.mapped_budget_id == budget_id)

        return query.scalar() or 0

    def get_period_costs(
        self,
        start_date: date,
        end_date: date,
        budget_id: Optional[int] = None,
        as_of_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Get costs aggregated by time period with actual/forecast split.

        Implements the temporal bucketing rules from the spec:
        - Past periods show actuals only
        - Current period shows blended (actual + remaining forecast)
        - Future periods show forecast only

        Args:
            start_date: Start of range
            end_date: End of range
            budget_id: Optional budget filter
            as_of_date: Reference date for actual/forecast split (default: today)

        Returns:
            List of period dicts with type, amounts
        """
        if as_of_date is None:
            as_of_date = date.today()

        costs = self.get_by_date_range(start_date, end_date, budget_id)

        # Group by month
        period_costs = defaultdict(lambda: {'actual': 0, 'count': 0})
        for cost in costs:
            period_key = cost.transaction_date.strftime('%Y-%m')
            period_costs[period_key]['actual'] += cost.gross_amount_cents
            period_costs[period_key]['count'] += 1

        results = []
        current_period = as_of_date.strftime('%Y-%m')

        for period_key, data in sorted(period_costs.items()):
            if period_key < current_period:
                period_type = 'actual'
            elif period_key == current_period:
                period_type = 'blended'
            else:
                period_type = 'forecast'

            results.append({
                'period_id': period_key,
                'type': period_type,
                'actual_cents': data['actual'] if period_type != 'forecast' else 0,
                'forecast_cents': 0,  # Would be populated by forecast service
                'total_cents': data['actual'],
                'transaction_count': data['count']
            })

        return results

    def get_summary_stats(self, budget_id: Optional[int] = None) -> Dict:
        """
        Get summary statistics for direct costs.

        Args:
            budget_id: Optional budget filter

        Returns:
            Dict with count, total, min, max, avg
        """
        query = self.session.query(DirectCostEntity)

        if budget_id:
            query = query.filter(DirectCostEntity.mapped_budget_id == budget_id)

        costs = query.all()

        if not costs:
            return {
                'count': 0,
                'total_cents': 0,
                'min_cents': 0,
                'max_cents': 0,
                'avg_cents': 0,
                'earliest_date': None,
                'latest_date': None
            }

        amounts = [c.gross_amount_cents for c in costs]
        dates = [c.transaction_date for c in costs if c.transaction_date]

        return {
            'count': len(costs),
            'total_cents': sum(amounts),
            'min_cents': min(amounts),
            'max_cents': max(amounts),
            'avg_cents': sum(amounts) // len(amounts) if amounts else 0,
            'earliest_date': min(dates).isoformat() if dates else None,
            'latest_date': max(dates).isoformat() if dates else None
        }

"""
Cost Aggregation Service - Handles vertical and horizontal reconciliation.

Implements the aggregation rules from the spec:
- Vertical: DirectCost → Budget → GMP reconciliation
- Horizontal: Time period consistency
- Temporal bucketing: Weekly and monthly aggregations
"""
from datetime import date, timedelta
from decimal import Decimal
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.models import GMP, BudgetEntity, DirectCostEntity
from app.infrastructure.repositories import (
    GMPRepository,
    BudgetRepository,
    DirectCostRepository,
)
from app.domain.exceptions import (
    GMPNotFoundError,
    BudgetNotFoundError,
    ReconciliationError,
    InvariantViolationError,
)


class CostAggregationService:
    """
    Service for cost aggregation and reconciliation.

    Ensures mathematical invariants:
    - Budget.actual_cost = Σ(DirectCost.amount | mapped_budget_id = Budget.id)
    - GMP.total_actual = Σ(Budget.actual_cost | gmp_id = GMP.id)
    - CumulativeActual(t) = Σ(DirectCost.amount | transaction_date <= t)
    """

    def __init__(self, session: Session):
        self.session = session
        self.gmp_repo = GMPRepository(session)
        self.budget_repo = BudgetRepository(session)
        self.cost_repo = DirectCostRepository(session)

    # =========================================================================
    # Vertical Reconciliation
    # =========================================================================

    def recalculate_budget_actual(self, budget_id: int) -> int:
        """
        Recalculate and return actual cost for a budget.

        This is a computed value - not stored but derived from direct costs.

        Args:
            budget_id: Budget identifier

        Returns:
            Total actual cost in cents
        """
        return self.cost_repo.get_total_by_budget(budget_id)

    def recalculate_gmp_totals(self, gmp_id: int) -> Dict:
        """
        Recalculate all totals for a GMP.

        Args:
            gmp_id: GMP identifier

        Returns:
            Dict with total_budgeted, total_actual, remaining
        """
        gmp = self.gmp_repo.get_by_id(gmp_id)
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))

        total_budgeted = self.gmp_repo.get_total_budgeted(gmp_id)
        total_actual = self.gmp_repo.get_total_actual(gmp_id)
        authorized = self.gmp_repo.get_authorized_amount(gmp_id)

        return {
            'gmp_id': gmp_id,
            'authorized_cents': authorized,
            'total_budgeted_cents': total_budgeted,
            'total_actual_cents': total_actual,
            'remaining_cents': authorized - total_actual,
            'budget_utilization_pct': (total_budgeted / authorized * 100) if authorized > 0 else 0,
            'spent_pct': (total_actual / authorized * 100) if authorized > 0 else 0,
        }

    def get_unmapped_total(self) -> int:
        """
        Get total of unmapped direct costs.

        Returns:
            Sum of costs not assigned to any budget
        """
        return self.cost_repo.get_total_unmapped()

    def get_full_reconciliation(self, gmp_id: int) -> Dict:
        """
        Get complete reconciliation for a GMP with all levels.

        Returns hierarchical structure:
        GMP -> List[Budget -> List[DirectCost]]

        Args:
            gmp_id: GMP identifier

        Returns:
            Complete reconciliation tree
        """
        gmp = self.gmp_repo.get_by_id(gmp_id)
        if not gmp:
            raise GMPNotFoundError(str(gmp_id))

        budgets = self.budget_repo.get_by_gmp(gmp_id)

        budget_details = []
        for budget in budgets:
            costs = self.cost_repo.get_by_budget(budget.id)
            actual = sum(c.gross_amount_cents for c in costs)

            budget_details.append({
                'id': budget.id,
                'uuid': budget.uuid,
                'cost_code': budget.cost_code,
                'zone': budget.zone,
                'description': budget.description,
                'current_budget_cents': budget.current_budget_cents,
                'committed_cents': budget.committed_cents,
                'actual_cost_cents': actual,
                'remaining_cents': budget.current_budget_cents - actual,
                'percent_spent': (actual / budget.current_budget_cents * 100) if budget.current_budget_cents > 0 else 0,
                'direct_costs': [
                    {
                        'id': c.id,
                        'uuid': c.uuid,
                        'vendor_name': c.vendor_name,
                        'description': c.description,
                        'transaction_date': c.transaction_date.isoformat() if c.transaction_date else None,
                        'gross_amount_cents': c.gross_amount_cents,
                        'payable_amount_cents': c.payable_amount_cents,
                    }
                    for c in costs
                ]
            })

        gmp_totals = self.recalculate_gmp_totals(gmp_id)

        return {
            'gmp': {
                'id': gmp.id,
                'uuid': gmp.uuid,
                'division': gmp.division,
                'zone': gmp.zone,
                'description': gmp.description,
                'original_amount_cents': gmp.original_amount_cents,
                **gmp_totals
            },
            'budgets': budget_details,
            'unmapped_total_cents': self.get_unmapped_total(),
        }

    def validate_vertical_reconciliation(self, gmp_id: int) -> Tuple[bool, List[str]]:
        """
        Validate vertical reconciliation invariants.

        Checks:
        - Each budget actual equals sum of its direct costs
        - GMP total equals sum of budget actuals

        Args:
            gmp_id: GMP identifier

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        gmp = self.gmp_repo.get_by_id(gmp_id)
        if not gmp:
            return False, [f"GMP {gmp_id} not found"]

        budgets = self.budget_repo.get_by_gmp(gmp_id)
        calculated_gmp_total = 0

        for budget in budgets:
            # Check budget actual matches sum of direct costs
            calculated_actual = self.cost_repo.get_total_by_budget(budget.id)
            calculated_gmp_total += calculated_actual

        # Check GMP total matches sum of budget actuals
        gmp_actual = self.gmp_repo.get_total_actual(gmp_id)
        if gmp_actual != calculated_gmp_total:
            errors.append(
                f"GMP total actual ({gmp_actual}) does not match "
                f"sum of budget actuals ({calculated_gmp_total})"
            )

        return len(errors) == 0, errors

    # =========================================================================
    # Horizontal Reconciliation (Temporal)
    # =========================================================================

    def get_cumulative_actual(
        self,
        as_of_date: date,
        budget_id: Optional[int] = None,
        gmp_id: Optional[int] = None
    ) -> int:
        """
        Get cumulative actual cost up to a date.

        Implements: CumulativeActual(t) = Σ(DirectCost.amount | transaction_date <= t)

        Args:
            as_of_date: Date to calculate through
            budget_id: Optional budget filter
            gmp_id: Optional GMP filter

        Returns:
            Cumulative cost in cents
        """
        if budget_id:
            return self.cost_repo.get_cumulative_by_date(as_of_date, budget_id)

        if gmp_id:
            budgets = self.budget_repo.get_by_gmp(gmp_id)
            return sum(
                self.cost_repo.get_cumulative_by_date(as_of_date, b.id)
                for b in budgets
            )

        # All costs
        return self.cost_repo.get_cumulative_by_date(as_of_date)

    def get_period_actual(
        self,
        start_date: date,
        end_date: date,
        budget_id: Optional[int] = None,
        gmp_id: Optional[int] = None
    ) -> int:
        """
        Get actual cost for a period.

        Implements: PeriodActual(t1, t2) = Σ(DirectCost.amount | t1 <= transaction_date <= t2)

        Args:
            start_date: Period start
            end_date: Period end
            budget_id: Optional budget filter
            gmp_id: Optional GMP filter

        Returns:
            Period cost in cents
        """
        costs = self.cost_repo.get_by_date_range(start_date, end_date, budget_id)

        if gmp_id:
            budget_ids = {b.id for b in self.budget_repo.get_by_gmp(gmp_id)}
            costs = [c for c in costs if c.mapped_budget_id in budget_ids]

        return sum(c.gross_amount_cents for c in costs)

    # =========================================================================
    # Temporal Bucketing
    # =========================================================================

    def get_weekly_totals(
        self,
        start_date: date,
        end_date: date,
        budget_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get costs aggregated by ISO week.

        Args:
            start_date: Range start
            end_date: Range end
            budget_id: Optional budget filter

        Returns:
            List of weekly totals with period info
        """
        costs = self.cost_repo.get_by_date_range(start_date, end_date, budget_id)

        weekly_totals = defaultdict(lambda: {'amount_cents': 0, 'count': 0})
        for cost in costs:
            if cost.transaction_date:
                iso = cost.transaction_date.isocalendar()
                week_key = f"{iso[0]}-W{iso[1]:02d}"
                weekly_totals[week_key]['amount_cents'] += cost.gross_amount_cents
                weekly_totals[week_key]['count'] += 1

        results = []
        for week_key in sorted(weekly_totals.keys()):
            data = weekly_totals[week_key]
            # Parse week key to get date range
            year, week = week_key.split('-W')
            week_start = date.fromisocalendar(int(year), int(week), 1)
            week_end = week_start + timedelta(days=6)

            results.append({
                'period_id': week_key,
                'period_start': week_start.isoformat(),
                'period_end': week_end.isoformat(),
                'iso_week': int(week),
                'iso_year': int(year),
                'amount_cents': data['amount_cents'],
                'transaction_count': data['count'],
            })

        return results

    def get_monthly_totals(
        self,
        start_date: date,
        end_date: date,
        budget_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get costs aggregated by calendar month.

        Args:
            start_date: Range start
            end_date: Range end
            budget_id: Optional budget filter

        Returns:
            List of monthly totals with period info
        """
        period_totals = self.cost_repo.get_period_totals(budget_id, 'month')

        # Filter to date range and format
        results = []
        for period_key in sorted(period_totals.keys()):
            year, month = period_key.split('-')
            period_start = date(int(year), int(month), 1)

            # Calculate period end (last day of month)
            if int(month) == 12:
                period_end = date(int(year) + 1, 1, 1) - timedelta(days=1)
            else:
                period_end = date(int(year), int(month) + 1, 1) - timedelta(days=1)

            # Filter to range
            if period_end < start_date or period_start > end_date:
                continue

            results.append({
                'period_id': period_key,
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat(),
                'year': int(year),
                'month': int(month),
                'amount_cents': period_totals[period_key],
            })

        return results

    def prorate_spanning_week(
        self,
        week_start: date,
        week_end: date,
        week_amount_cents: int
    ) -> Dict[str, int]:
        """
        Prorate a week's cost across months it spans.

        Implements the SPANNING_WEEK_PRORATION rule from the spec.

        Example: Week Jan 28 - Feb 3 (4 days Jan, 3 days Feb)
        Returns: {'2024-01': 4/7 * amount, '2024-02': 3/7 * amount}

        Args:
            week_start: First day of week
            week_end: Last day of week
            week_amount_cents: Total week amount in cents

        Returns:
            Dict mapping month key to prorated amount
        """
        days_per_month = defaultdict(int)
        current = week_start

        while current <= week_end:
            month_key = current.strftime('%Y-%m')
            days_per_month[month_key] += 1
            current += timedelta(days=1)

        total_days = sum(days_per_month.values())

        prorated = {}
        for month_key, days in days_per_month.items():
            prorated[month_key] = int(week_amount_cents * days / total_days)

        # Handle rounding - add remainder to largest month
        remainder = week_amount_cents - sum(prorated.values())
        if remainder != 0:
            largest_month = max(days_per_month.keys(), key=lambda k: days_per_month[k])
            prorated[largest_month] += remainder

        return prorated

    # =========================================================================
    # Period Analysis with Actuals/Forecast Split
    # =========================================================================

    def get_period_breakdown(
        self,
        start_date: date,
        end_date: date,
        as_of_date: date,
        budget_id: Optional[int] = None,
        granularity: str = 'month'
    ) -> List[Dict]:
        """
        Get period breakdown with actual/forecast split.

        Implements the PAST_PERIODS_SHOW_ACTUALS and CURRENT_PERIOD_BLENDED rules.

        Args:
            start_date: Range start
            end_date: Range end
            as_of_date: Reference date for actual/forecast split
            budget_id: Optional budget filter
            granularity: 'week' or 'month'

        Returns:
            List of periods with type (actual/forecast/blended) and amounts
        """
        if granularity == 'week':
            raw_totals = self.get_weekly_totals(start_date, end_date, budget_id)
        else:
            raw_totals = self.get_monthly_totals(start_date, end_date, budget_id)

        current_period = as_of_date.strftime('%Y-%m') if granularity == 'month' else \
            f"{as_of_date.isocalendar()[0]}-W{as_of_date.isocalendar()[1]:02d}"

        results = []
        for period in raw_totals:
            period_id = period['period_id']

            if period_id < current_period:
                period_type = 'actual'
                actual_cents = period['amount_cents']
                forecast_cents = 0
            elif period_id == current_period:
                period_type = 'blended'
                actual_cents = period['amount_cents']
                forecast_cents = 0  # Would be populated by forecast service
            else:
                period_type = 'forecast'
                actual_cents = 0
                forecast_cents = 0  # Would be populated by forecast service

            results.append({
                **period,
                'type': period_type,
                'actual_cents': actual_cents,
                'forecast_cents': forecast_cents,
                'total_cents': actual_cents + forecast_cents,
            })

        return results

    # =========================================================================
    # Cascade Update Handlers
    # =========================================================================

    def handle_direct_cost_change(
        self,
        cost_id: int,
        operation: str,
        old_budget_id: Optional[int] = None,
        new_budget_id: Optional[int] = None
    ) -> Dict:
        """
        Handle cascading updates when a direct cost changes.

        Recalculates affected budget and GMP totals.

        Args:
            cost_id: Direct cost identifier
            operation: 'insert', 'update', or 'delete'
            old_budget_id: Previous budget mapping (for updates)
            new_budget_id: New budget mapping (for inserts/updates)

        Returns:
            Dict with affected entities and new totals
        """
        affected = {
            'budgets': [],
            'gmps': []
        }

        budget_ids = set()
        if old_budget_id:
            budget_ids.add(old_budget_id)
        if new_budget_id:
            budget_ids.add(new_budget_id)

        for budget_id in budget_ids:
            budget = self.budget_repo.get_by_id(budget_id)
            if budget:
                actual = self.recalculate_budget_actual(budget_id)
                affected['budgets'].append({
                    'budget_id': budget_id,
                    'actual_cost_cents': actual,
                })

                # Cascade to GMP
                gmp_totals = self.recalculate_gmp_totals(budget.gmp_id)
                if gmp_totals not in [a for a in affected['gmps'] if a['gmp_id'] == budget.gmp_id]:
                    affected['gmps'].append(gmp_totals)

        return affected

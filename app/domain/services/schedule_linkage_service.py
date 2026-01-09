"""
Schedule Linkage Service - Connects schedule to cost hierarchy.

Implements earned value management formulas and schedule-driven forecasting:
- Planned Value (PV)
- Earned Value (EV)
- Schedule Variance (SV)
- Schedule Performance Index (SPI)
"""
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import (
    ScheduleActivity,
    ScheduleToGMPMapping,
    GMP,
    BudgetEntity,
    DirectCostEntity,
)
from app.infrastructure.repositories import (
    ScheduleRepository,
    GMPRepository,
    BudgetRepository,
    DirectCostRepository,
)
from app.domain.exceptions import (
    ScheduleNotFoundError,
    GMPNotFoundError,
)


class ScheduleLinkageService:
    """
    Service for schedule-cost integration.

    The schedule is the PRIMARY DRIVER of cost timing expectations.
    This service calculates earned value metrics and schedule-driven forecasts.
    """

    def __init__(self, session: Session):
        self.session = session
        self.schedule_repo = ScheduleRepository(session)
        self.gmp_repo = GMPRepository(session)
        self.budget_repo = BudgetRepository(session)
        self.cost_repo = DirectCostRepository(session)

    # =========================================================================
    # Earned Value Calculations
    # =========================================================================

    def calculate_planned_value(
        self,
        gmp_division: str,
        as_of_date: Optional[date] = None
    ) -> Dict:
        """
        Calculate Planned Value (PV) for a GMP division.

        PV = Budget × Schedule.expected_pct_complete(t)
        "Budgeted cost of work scheduled"

        Args:
            gmp_division: GMP division name
            as_of_date: Reference date (default: today)

        Returns:
            Dict with PV calculation details
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Get total budget for this division
        budget_total = self._get_division_budget(gmp_division)

        # Get activities linked to this division
        activities = self.schedule_repo.get_by_gmp_division(gmp_division)

        if not activities:
            return {
                'gmp_division': gmp_division,
                'budget_at_completion_cents': budget_total,
                'planned_value_cents': 0,
                'expected_pct_complete': 0,
                'activity_count': 0,
            }

        # Calculate weighted expected progress
        total_weight = 0
        weighted_progress = 0

        for activity in activities:
            # Get mapping weight
            mappings = self.schedule_repo.get_gmp_mappings(activity.id)
            weight = sum(m.weight for m in mappings if m.gmp_division == gmp_division)
            total_weight += weight

            # Calculate expected progress based on dates
            expected_pct = self._calculate_expected_progress(activity, as_of_date)
            weighted_progress += expected_pct * weight

        if total_weight > 0:
            overall_expected = weighted_progress / total_weight
        else:
            overall_expected = 0

        pv_cents = int(budget_total * overall_expected)

        return {
            'gmp_division': gmp_division,
            'budget_at_completion_cents': budget_total,
            'planned_value_cents': pv_cents,
            'expected_pct_complete': round(overall_expected * 100, 2),
            'activity_count': len(activities),
            'as_of_date': as_of_date.isoformat(),
        }

    def calculate_earned_value(
        self,
        gmp_division: str,
        as_of_date: Optional[date] = None
    ) -> Dict:
        """
        Calculate Earned Value (EV) for a GMP division.

        EV = Budget × Actual.pct_complete(t)
        "Budgeted cost of work performed"

        Args:
            gmp_division: GMP division name
            as_of_date: Reference date (default: today)

        Returns:
            Dict with EV calculation details
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Get total budget for this division
        budget_total = self._get_division_budget(gmp_division)

        # Get activities linked to this division
        activities = self.schedule_repo.get_by_gmp_division(gmp_division)

        if not activities:
            return {
                'gmp_division': gmp_division,
                'budget_at_completion_cents': budget_total,
                'earned_value_cents': 0,
                'actual_pct_complete': 0,
                'activity_count': 0,
            }

        # Calculate weighted actual progress
        total_weight = 0
        weighted_progress = 0

        for activity in activities:
            # Get mapping weight
            mappings = self.schedule_repo.get_gmp_mappings(activity.id)
            weight = sum(m.weight for m in mappings if m.gmp_division == gmp_division)
            total_weight += weight

            # Use actual progress from activity
            actual_pct = activity.progress_pct or 0
            weighted_progress += actual_pct * weight

        if total_weight > 0:
            overall_actual = weighted_progress / total_weight
        else:
            overall_actual = 0

        ev_cents = int(budget_total * overall_actual)

        return {
            'gmp_division': gmp_division,
            'budget_at_completion_cents': budget_total,
            'earned_value_cents': ev_cents,
            'actual_pct_complete': round(overall_actual * 100, 2),
            'activity_count': len(activities),
            'as_of_date': as_of_date.isoformat(),
        }

    def calculate_actual_cost(
        self,
        gmp_division: str,
        as_of_date: Optional[date] = None
    ) -> Dict:
        """
        Calculate Actual Cost (AC) for a GMP division.

        AC = Σ(DirectCost.amount | transaction_date <= t)
        "Actual cost of work performed"

        Args:
            gmp_division: GMP division name
            as_of_date: Reference date (default: today)

        Returns:
            Dict with AC calculation details
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Get GMPs for this division
        gmps = self.gmp_repo.get_by_division(gmp_division)
        total_actual = 0
        transaction_count = 0

        for gmp in gmps:
            budgets = self.budget_repo.get_by_gmp(gmp.id)
            for budget in budgets:
                costs = self.session.query(DirectCostEntity).filter(
                    DirectCostEntity.mapped_budget_id == budget.id,
                    DirectCostEntity.transaction_date <= as_of_date
                ).all()
                total_actual += sum(c.gross_amount_cents for c in costs)
                transaction_count += len(costs)

        return {
            'gmp_division': gmp_division,
            'actual_cost_cents': total_actual,
            'transaction_count': transaction_count,
            'as_of_date': as_of_date.isoformat(),
        }

    def calculate_full_evm(
        self,
        gmp_division: str,
        as_of_date: Optional[date] = None
    ) -> Dict:
        """
        Calculate full Earned Value Management metrics.

        Includes:
        - PV (Planned Value)
        - EV (Earned Value)
        - AC (Actual Cost)
        - SV (Schedule Variance) = EV - PV
        - CV (Cost Variance) = EV - AC
        - SPI (Schedule Performance Index) = EV / PV
        - CPI (Cost Performance Index) = EV / AC

        Args:
            gmp_division: GMP division name
            as_of_date: Reference date

        Returns:
            Dict with all EVM metrics
        """
        if as_of_date is None:
            as_of_date = date.today()

        pv_data = self.calculate_planned_value(gmp_division, as_of_date)
        ev_data = self.calculate_earned_value(gmp_division, as_of_date)
        ac_data = self.calculate_actual_cost(gmp_division, as_of_date)

        bac = pv_data['budget_at_completion_cents']
        pv = pv_data['planned_value_cents']
        ev = ev_data['earned_value_cents']
        ac = ac_data['actual_cost_cents']

        # Calculate variances
        sv = ev - pv  # Schedule Variance
        cv = ev - ac  # Cost Variance

        # Calculate indices
        spi = ev / pv if pv > 0 else 0  # Schedule Performance Index
        cpi = ev / ac if ac > 0 else 0  # Cost Performance Index

        # EAC calculations
        # EAC = BAC / CPI (assuming performance continues)
        eac_cpi = int(bac / cpi) if cpi > 0 else bac
        # EAC = AC + (BAC - EV) (original estimate for remaining work)
        eac_remaining = ac + (bac - ev)
        # EAC = AC + (BAC - EV) / (CPI * SPI) (considering both indices)
        eac_combined = int(ac + (bac - ev) / (cpi * spi)) if (cpi > 0 and spi > 0) else bac

        return {
            'gmp_division': gmp_division,
            'as_of_date': as_of_date.isoformat(),

            # Base metrics
            'bac_cents': bac,
            'pv_cents': pv,
            'ev_cents': ev,
            'ac_cents': ac,

            # Progress percentages
            'expected_pct_complete': pv_data['expected_pct_complete'],
            'actual_pct_complete': ev_data['actual_pct_complete'],

            # Variances
            'sv_cents': sv,
            'cv_cents': cv,
            'sv_interpretation': 'ahead' if sv > 0 else ('behind' if sv < 0 else 'on_schedule'),
            'cv_interpretation': 'under_budget' if cv > 0 else ('over_budget' if cv < 0 else 'on_budget'),

            # Indices
            'spi': round(spi, 3),
            'cpi': round(cpi, 3),
            'spi_interpretation': 'ahead' if spi > 1 else ('behind' if spi < 1 else 'on_schedule'),
            'cpi_interpretation': 'under_budget' if cpi > 1 else ('over_budget' if cpi < 1 else 'on_budget'),

            # EAC projections
            'eac_cpi_cents': eac_cpi,
            'eac_remaining_cents': eac_remaining,
            'eac_combined_cents': eac_combined,

            # ETC (Estimate to Complete)
            'etc_cents': max(0, bac - ev),

            # VAC (Variance at Completion)
            'vac_cents': bac - eac_cpi,
        }

    # =========================================================================
    # Schedule-Driven Forecasting
    # =========================================================================

    def get_schedule_based_forecast(
        self,
        gmp_division: str,
        start_date: date,
        end_date: date,
        granularity: str = 'month'
    ) -> List[Dict]:
        """
        Generate schedule-based cost forecast.

        Distributes remaining budget across future periods based on
        scheduled activity timing.

        Args:
            gmp_division: GMP division name
            start_date: Forecast start
            end_date: Forecast end
            granularity: 'week' or 'month'

        Returns:
            List of period forecasts
        """
        activities = self.schedule_repo.get_by_gmp_division(gmp_division)
        budget_total = self._get_division_budget(gmp_division)
        actual_total = self.calculate_actual_cost(gmp_division)['actual_cost_cents']
        remaining = max(0, budget_total - actual_total)

        if not activities or remaining == 0:
            return []

        # Calculate work distribution by period
        period_weights = self._calculate_period_weights(
            activities, start_date, end_date, granularity
        )

        # Distribute remaining budget
        results = []
        for period_key, weight in sorted(period_weights.items()):
            forecast_cents = int(remaining * weight)

            results.append({
                'period_id': period_key,
                'forecast_cents': forecast_cents,
                'weight': round(weight, 4),
                'granularity': granularity,
            })

        return results

    def _calculate_period_weights(
        self,
        activities: List[ScheduleActivity],
        start_date: date,
        end_date: date,
        granularity: str
    ) -> Dict[str, float]:
        """
        Calculate work distribution weights by period.

        Weights are based on remaining work in each activity
        distributed across its remaining duration.
        """
        period_work = {}

        for activity in activities:
            if not activity.start_date or not activity.finish_date:
                continue

            # Calculate remaining work
            remaining_pct = 1.0 - (activity.progress_pct or 0)
            if remaining_pct <= 0:
                continue

            # Calculate days in each period
            act_start = max(activity.start_date, start_date)
            act_end = min(activity.finish_date, end_date)

            if act_start > act_end:
                continue

            current = act_start
            while current <= act_end:
                if granularity == 'week':
                    iso = current.isocalendar()
                    period_key = f"{iso[0]}-W{iso[1]:02d}"
                else:
                    period_key = current.strftime('%Y-%m')

                if period_key not in period_work:
                    period_work[period_key] = 0

                # Add proportional work for this day
                total_days = (activity.finish_date - activity.start_date).days + 1
                day_weight = remaining_pct / total_days if total_days > 0 else 0
                period_work[period_key] += day_weight

                current += timedelta(days=1)

        # Normalize to sum to 1.0
        total_weight = sum(period_work.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in period_work.items()}
        return {}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_division_budget(self, gmp_division: str) -> int:
        """Get total budget for a GMP division."""
        gmps = self.gmp_repo.get_by_division(gmp_division)
        total = 0
        for gmp in gmps:
            budgets = self.budget_repo.get_by_gmp(gmp.id)
            total += sum(b.current_budget_cents for b in budgets)
        return total

    def _calculate_expected_progress(
        self,
        activity: ScheduleActivity,
        as_of_date: date
    ) -> float:
        """
        Calculate expected progress for an activity at a given date.

        Based on linear interpolation between start and finish dates.
        """
        if not activity.start_date or not activity.finish_date:
            return 0

        if as_of_date >= activity.finish_date:
            return 1.0
        if as_of_date <= activity.start_date:
            return 0.0

        elapsed = (as_of_date - activity.start_date).days
        total = (activity.finish_date - activity.start_date).days

        return elapsed / total if total > 0 else 0

    # =========================================================================
    # Forecast Recalculation
    # =========================================================================

    def recalculate_all_forecasts(self, project_id: Optional[int] = None) -> Dict:
        """
        Recalculate forecasts for all (or project) GMP divisions.

        Triggered when schedule changes.

        Args:
            project_id: Optional project filter

        Returns:
            Dict with recalculation results
        """
        results = {
            'divisions_updated': 0,
            'divisions': []
        }

        # Get all unique divisions from mappings
        mappings = self.session.query(
            ScheduleToGMPMapping.gmp_division
        ).distinct().all()

        for (division,) in mappings:
            evm = self.calculate_full_evm(division)
            results['divisions'].append({
                'gmp_division': division,
                'eac_cents': evm['eac_cpi_cents'],
                'spi': evm['spi'],
                'cpi': evm['cpi'],
            })
            results['divisions_updated'] += 1

        return results

    def update_schedule_actuals(
        self,
        affected_dates: List[date]
    ) -> Dict:
        """
        Update schedule-related calculations for affected dates.

        Called when direct costs change.

        Args:
            affected_dates: Dates where costs changed

        Returns:
            Dict with update results
        """
        # Find activities active on affected dates
        affected_activities = set()
        for target_date in affected_dates:
            activities = self.schedule_repo.get_active_on_date(target_date)
            for activity in activities:
                affected_activities.add(activity.id)

        # Get affected GMP divisions
        affected_divisions = set()
        for activity_id in affected_activities:
            mappings = self.schedule_repo.get_gmp_mappings(activity_id)
            for mapping in mappings:
                affected_divisions.add(mapping.gmp_division)

        return {
            'affected_activity_count': len(affected_activities),
            'affected_division_count': len(affected_divisions),
            'divisions': list(affected_divisions),
        }

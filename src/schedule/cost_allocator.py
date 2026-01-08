"""
Activity Cost Allocator - Distributes GMP budget across activities

Each activity gets an EXPECTED cost based on:
1. Trade GMP budget
2. Activity duration relative to trade's total duration
3. Activity type cost intensity
4. Cost curve shape for the phase

Formula:
    activity_expected_cost = (
        trade_gmp_budget
        * (activity_duration / trade_total_duration)
        * activity_intensity_factor
        * cost_weight
    )
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .parser import ScheduleParser, Activity, Phase, CostCurve

logger = logging.getLogger(__name__)


@dataclass
class ActivityCostAllocation:
    """Expected cost for an activity"""
    activity: Activity
    expected_cost: float
    cost_curve_value: float
    trade_budget: float
    daily_burn_rate: float


class ActivityCostAllocator:
    """
    Allocates GMP budget to individual activities.
    This creates the EXPECTED cost profile based on schedule.

    Variance = Actual Cost - Expected Cost (based on schedule)
    """

    def __init__(
        self,
        schedule_parser: ScheduleParser,
        gmp_breakdown: pd.DataFrame
    ):
        """
        Initialize allocator.

        Args:
            schedule_parser: Parsed schedule with activities
            gmp_breakdown: GMP breakdown DataFrame with trade budgets
        """
        self.parser = schedule_parser
        self.gmp = self._parse_gmp(gmp_breakdown)
        self.allocations: Dict[str, List[ActivityCostAllocation]] = {}
        self.trade_totals: Dict[str, float] = {}

        self._allocate_costs()

    def _parse_gmp(self, gmp_df: pd.DataFrame) -> Dict[str, float]:
        """Parse GMP breakdown into trade -> budget dict"""
        gmp = {}

        # Try different column names
        trade_cols = ['trade_category', 'Cost Code - Description', 'Trade', 'Description']
        amount_cols = ['gmp_total', 'GMP SOV', 'Total', 'Amount', 'Budget']

        trade_col = None
        amount_col = None

        for col in trade_cols:
            if col in gmp_df.columns:
                trade_col = col
                break

        for col in amount_cols:
            if col in gmp_df.columns:
                amount_col = col
                break

        if not trade_col or not amount_col:
            logger.warning(f"Could not find trade/amount columns in GMP breakdown")
            return gmp

        for _, row in gmp_df.iterrows():
            trade = str(row.get(trade_col, '')).strip()
            amount = row.get(amount_col, 0)

            # Parse currency if needed
            if isinstance(amount, str):
                amount = amount.replace('$', '').replace(',', '').strip()
                try:
                    amount = float(amount) if amount else 0
                except ValueError:
                    amount = 0

            if trade and float(amount) > 0:
                gmp[trade] = float(amount)

        logger.info(f"Parsed {len(gmp)} trades from GMP breakdown, total: ${sum(gmp.values()):,.0f}")
        return gmp

    def _get_cost_curve_value(self, curve: CostCurve, t: float) -> float:
        """
        Get cost curve intensity at time t (0 to 1).

        Returns a multiplier indicating cost intensity at this point in the curve.
        """
        t = max(0, min(1, t))

        if curve == CostCurve.STEP:
            return 1.0 if t > 0 else 0.0
        elif curve == CostCurve.FRONT_LOADED:
            # Heavy early: derivative of 1 - (1-t)^2 = 2(1-t)
            return 2 * (1 - t)
        elif curve == CostCurve.CONCENTRATED:
            # S-curve derivative: 6t - 6t^2 = 6t(1-t)
            return 6 * t * (1 - t)
        elif curve == CostCurve.NORMAL:
            # Bell curve: sin(pi*t) derivative = pi*cos(pi*t)
            return max(0, np.cos(np.pi * t) * np.pi / 2)
        elif curve == CostCurve.EXTENDED:
            # Linear: constant intensity
            return 1.0
        elif curve == CostCurve.BACK_LOADED:
            # Heavy late: derivative of t^2 = 2t
            return 2 * t
        else:
            return 1.0

    def _allocate_costs(self):
        """Allocate GMP budget to activities"""

        # Group activities by trade
        trade_activities: Dict[str, List[Activity]] = {}

        for activity in self.parser.activities:
            trade = activity.primary_trade
            if trade not in trade_activities:
                trade_activities[trade] = []
            trade_activities[trade].append(activity)

        # Allocate budget to each trade's activities
        for trade, activities in trade_activities.items():
            trade_budget = self.gmp.get(trade, 0)

            if trade_budget == 0 or not activities:
                continue

            # Calculate total weighted duration for the trade
            total_weighted_duration = sum(
                a.duration_days * a.cost_weight * a.intensity_factor
                for a in activities
            )

            if total_weighted_duration == 0:
                continue

            # Allocate to each activity
            allocations = []
            allocated_total = 0

            for activity in activities:
                # Proportion of trade budget based on duration, weight, and intensity
                weighted_duration = activity.duration_days * activity.cost_weight * activity.intensity_factor
                proportion = weighted_duration / total_weighted_duration
                expected_cost = trade_budget * proportion

                # Get cost curve adjustment for intensity weighting
                phase = self._get_activity_phase(activity)
                curve_value = 1.0
                if phase:
                    mid_point = activity.start + timedelta(days=activity.duration_days / 2)
                    curve_value = self._get_cost_curve_value(
                        phase.cost_curve,
                        phase.pct_complete(mid_point)
                    )
                    # Normalize curve value (don't want it to change total, just distribution)
                    curve_value = max(0.5, min(2.0, curve_value))

                # Daily burn rate
                daily_burn = expected_cost / max(1, activity.duration_days)

                allocations.append(ActivityCostAllocation(
                    activity=activity,
                    expected_cost=expected_cost,
                    cost_curve_value=curve_value,
                    trade_budget=trade_budget,
                    daily_burn_rate=daily_burn
                ))

                allocated_total += expected_cost

            self.allocations[trade] = allocations
            self.trade_totals[trade] = allocated_total

        logger.info(f"Allocated costs to {sum(len(a) for a in self.allocations.values())} "
                   f"activities across {len(self.allocations)} trades")

    def _get_activity_phase(self, activity: Activity) -> Optional[Phase]:
        """Get the phase containing this activity"""
        for phase in self.parser.phases:
            if activity in phase.activities:
                return phase
        return None

    def get_expected_cost_to_date(
        self,
        trade: str,
        as_of: datetime
    ) -> float:
        """
        Get expected cumulative cost for a trade based on schedule.

        This is what we SHOULD have spent by this date if following the schedule.
        """
        if trade not in self.allocations:
            return 0.0

        total = 0.0

        for alloc in self.allocations[trade]:
            activity = alloc.activity

            if activity.is_complete(as_of):
                # Activity complete - full cost expected
                total += alloc.expected_cost
            elif activity.is_active(as_of):
                # Activity in progress - pro-rate based on elapsed time
                pct = activity.pct_complete(as_of)
                # Apply cost curve adjustment
                adjusted_pct = pct * alloc.cost_curve_value
                total += alloc.expected_cost * min(1.0, adjusted_pct)
            # Activity not started - no cost expected

        return total

    def get_expected_cost_this_period(
        self,
        trade: str,
        period_start: datetime,
        period_end: datetime
    ) -> float:
        """
        Get expected cost for a trade in a specific period.
        """
        if trade not in self.allocations:
            return 0.0

        total = 0.0

        for alloc in self.allocations[trade]:
            activity = alloc.activity

            # Check if activity overlaps with period
            overlap_start = max(activity.start, period_start)
            overlap_end = min(activity.finish, period_end)

            if overlap_start <= overlap_end:
                # Calculate proportion of activity in this period
                overlap_days = (overlap_end - overlap_start).days + 1
                activity_days = max(1, activity.duration_days)
                proportion = overlap_days / activity_days

                # Apply curve adjustment
                mid_overlap = overlap_start + timedelta(days=overlap_days / 2)
                phase = self._get_activity_phase(activity)
                if phase:
                    curve_mult = self._get_cost_curve_value(
                        phase.cost_curve,
                        phase.pct_complete(mid_overlap)
                    )
                    proportion *= curve_mult

                total += alloc.expected_cost * min(1.0, proportion)

        return total

    def get_schedule_variance(
        self,
        trade: str,
        actual_cost: float,
        as_of: datetime
    ) -> Tuple[float, float]:
        """
        Calculate schedule variance: actual - expected.

        Returns:
            (variance_amount, variance_percent)

        Positive = over-spending vs schedule
        Negative = under-spending vs schedule
        """
        expected = self.get_expected_cost_to_date(trade, as_of)
        variance = actual_cost - expected
        variance_pct = variance / expected if expected > 0 else 0

        return variance, variance_pct

    def get_remaining_budget(
        self,
        trade: str,
        actual_spent: float
    ) -> float:
        """Get remaining budget for a trade"""
        budget = self.gmp.get(trade, 0)
        return budget - actual_spent

    def get_expected_remaining_cost(
        self,
        trade: str,
        as_of: datetime
    ) -> float:
        """Get expected remaining cost based on incomplete activities"""
        if trade not in self.allocations:
            return 0.0

        remaining = 0.0

        for alloc in self.allocations[trade]:
            activity = alloc.activity

            if activity.is_complete(as_of):
                continue
            elif activity.is_active(as_of):
                # Remaining portion of active activity
                pct_remaining = 1.0 - activity.pct_complete(as_of)
                remaining += alloc.expected_cost * pct_remaining
            else:
                # Activity not started - full cost remaining
                remaining += alloc.expected_cost

        return remaining

    def get_forecast_at_completion(
        self,
        trade: str,
        actual_spent: float,
        as_of: datetime
    ) -> Tuple[float, float]:
        """
        Forecast total cost at completion based on current variance trend.

        Returns:
            (forecast_at_completion, variance_from_budget)
        """
        expected_to_date = self.get_expected_cost_to_date(trade, as_of)
        expected_remaining = self.get_expected_remaining_cost(trade, as_of)
        budget = self.gmp.get(trade, 0)

        if expected_to_date > 0:
            # Calculate cost performance index
            cpi = expected_to_date / actual_spent if actual_spent > 0 else 1.0
            # Forecast remaining with current performance
            forecast_remaining = expected_remaining / max(0.5, min(2.0, cpi))
        else:
            forecast_remaining = expected_remaining

        forecast_total = actual_spent + forecast_remaining
        variance_from_budget = forecast_total - budget

        return forecast_total, variance_from_budget

    def get_trade_schedule_summary(self, as_of: datetime) -> pd.DataFrame:
        """Get schedule-based summary for all trades"""
        records = []

        for trade, allocs in self.allocations.items():
            activities = [a.activity for a in allocs]
            budget = self.gmp.get(trade, 0)

            # Activity counts
            total_activities = len(activities)
            completed = len([a for a in activities if a.is_complete(as_of)])
            active = len([a for a in activities if a.is_active(as_of)])
            remaining = len([a for a in activities if not a.is_complete(as_of) and not a.is_active(as_of)])

            # Expected costs
            expected_to_date = self.get_expected_cost_to_date(trade, as_of)
            expected_remaining = self.get_expected_remaining_cost(trade, as_of)

            # Timeline
            if activities:
                earliest = min(a.start for a in activities)
                latest = max(a.finish for a in activities)
            else:
                earliest = latest = None

            records.append({
                'trade': trade,
                'gmp_budget': budget,
                'total_activities': total_activities,
                'completed_activities': completed,
                'active_activities': active,
                'remaining_activities': remaining,
                'expected_cost_to_date': expected_to_date,
                'expected_remaining': expected_remaining,
                'expected_pct_complete': expected_to_date / budget if budget > 0 else 0,
                'earliest_start': earliest,
                'latest_finish': latest
            })

        return pd.DataFrame(records)

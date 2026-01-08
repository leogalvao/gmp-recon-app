"""
Schedule-Driven Feature Builder

ALL features are indexed by schedule position, not just time.
The schedule is the PRIMARY driver - costs follow schedule.

Feature Hierarchy:
1. Level 1 - Schedule (PRIMARY): Activity timing, phase position, project progress
2. Level 2 - Trade Schedule: Trade's position in schedule
3. Level 3 - Cost: Actual cost patterns
4. Level 4 - Budget: Budget constraints
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

from ..schedule.parser import ScheduleParser, Activity, Phase
from ..schedule.cost_allocator import ActivityCostAllocator

logger = logging.getLogger(__name__)


@dataclass
class ScheduleDrivenFeatures:
    """Features for a single time period, driven by schedule"""

    # Time reference
    period: str  # year_month
    as_of_date: datetime

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 1: PROJECT SCHEDULE FEATURES (PRIMARY)
    # ─────────────────────────────────────────────────────────────────────────
    project_pct_complete: float
    project_days_elapsed: int
    project_days_remaining: int
    project_total_days: int

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 1: PHASE FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    current_phase: str
    phase_pct_complete: float
    phase_days_remaining: int
    phases_active_count: int

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 1: ACTIVITY FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    total_active_activities: int
    critical_path_activities: int
    activities_completed_to_date: int
    activities_remaining: int
    avg_float: float
    min_float: float

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 2: TRADE SCHEDULE FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    trade_activities_active: int
    trade_activities_complete: int
    trade_activities_remaining: int
    trade_phase_active: bool
    trade_expected_pct_complete: float
    trade_days_to_peak: int
    trade_days_since_start: int

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 3: COST FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    trade_actual_cost: float
    trade_cost_this_period: float
    trade_expected_cost: float
    trade_schedule_variance: float
    trade_variance_pct: float
    trade_cost_velocity: float  # Rolling avg cost per day

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 4: BUDGET FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    trade_gmp_budget: float
    trade_budget_remaining: float
    trade_actual_pct_spent: float
    trade_forecast_at_completion: float
    trade_budget_variance: float


class ScheduleDrivenFeatureBuilder:
    """
    Builds training features with SCHEDULE as the primary driver.

    The key insight: Schedule position predicts cost better than time alone.
    """

    def __init__(
        self,
        schedule_parser: ScheduleParser,
        cost_allocator: ActivityCostAllocator,
        gmp_breakdown: pd.DataFrame
    ):
        """
        Initialize feature builder.

        Args:
            schedule_parser: Parsed schedule with activities
            cost_allocator: Cost allocations to activities
            gmp_breakdown: GMP breakdown for budget context
        """
        self.parser = schedule_parser
        self.allocator = cost_allocator
        self.gmp = self._parse_gmp(gmp_breakdown)

    def _parse_gmp(self, gmp_df: pd.DataFrame) -> Dict[str, float]:
        """Parse GMP to dict"""
        result = {}

        trade_cols = ['trade_category', 'Cost Code - Description', 'Trade', 'Description']
        amount_cols = ['gmp_total', 'GMP SOV', 'Total', 'Amount', 'Budget']

        trade_col = next((c for c in trade_cols if c in gmp_df.columns), None)
        amount_col = next((c for c in amount_cols if c in gmp_df.columns), None)

        if not trade_col or not amount_col:
            return result

        for _, row in gmp_df.iterrows():
            trade = str(row.get(trade_col, '')).strip()
            amount = row.get(amount_col, 0)
            if isinstance(amount, str):
                amount = float(amount.replace('$', '').replace(',', '').strip() or 0)
            if trade:
                result[trade] = float(amount)

        return result

    def _get_trade_peak_date(self, trade_name: str) -> Optional[datetime]:
        """Get the date when trade activity is at its peak"""
        activities = self.parser.get_activities_for_trade(trade_name)
        if not activities:
            return None

        # Peak is middle of the trade's activity window
        earliest = min(a.start for a in activities)
        latest = max(a.finish for a in activities)
        return earliest + (latest - earliest) / 2

    def _get_trade_start_date(self, trade_name: str) -> Optional[datetime]:
        """Get when trade activities first start"""
        activities = self.parser.get_activities_for_trade(trade_name)
        if not activities:
            return None
        return min(a.start for a in activities)

    def build_features_for_period(
        self,
        trade_name: str,
        period: str,
        cumulative_actual_cost: float,
        period_cost: float = 0,
        cost_history: Optional[List[float]] = None
    ) -> ScheduleDrivenFeatures:
        """
        Build all features for a trade in a specific period.

        Args:
            trade_name: GMP trade name
            period: Year-month string (e.g., "2024-06")
            cumulative_actual_cost: Total spent to date
            period_cost: Cost for this period
            cost_history: Rolling history for velocity calculation
        """
        # Convert period to date (mid-month)
        try:
            period_dt = pd.Period(period, freq='M')
            as_of = period_dt.to_timestamp() + timedelta(days=15)
        except Exception:
            as_of = datetime.now()

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 1: PROJECT-LEVEL SCHEDULE
        # ─────────────────────────────────────────────────────────────────────
        project_pct = self.parser.project_pct_complete(as_of)

        if self.parser.project_start:
            days_elapsed = (as_of - self.parser.project_start).days
        else:
            days_elapsed = 0

        if self.parser.project_end:
            days_remaining = max(0, (self.parser.project_end - as_of).days)
            total_days = (self.parser.project_end - self.parser.project_start).days
        else:
            days_remaining = 0
            total_days = 0

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 1: PHASE-LEVEL
        # ─────────────────────────────────────────────────────────────────────
        current_phase = self.parser.get_current_phase(as_of)
        phase_id = current_phase.id if current_phase else 'NONE'
        phase_pct = current_phase.pct_complete(as_of) if current_phase else 0
        phase_remaining = (current_phase.end - as_of).days if current_phase else 0
        phases_active = len(self.parser.get_active_phases(as_of))

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 1: ACTIVITY-LEVEL
        # ─────────────────────────────────────────────────────────────────────
        active_activities = self.parser.get_active_activities(as_of)
        all_activities = self.parser.activities

        total_active = len(active_activities)
        critical = len([a for a in active_activities if a.is_critical])
        completed = len([a for a in all_activities if a.is_complete(as_of)])
        remaining = len([a for a in all_activities if not a.is_complete(as_of)])

        floats = [a.total_float for a in active_activities]
        avg_float = np.mean(floats) if floats else 0
        min_float = min(floats) if floats else 0

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 2: TRADE SCHEDULE
        # ─────────────────────────────────────────────────────────────────────
        trade_activities = self.parser.get_activities_for_trade(trade_name)
        trade_active = [a for a in trade_activities if a.is_active(as_of)]
        trade_complete = [a for a in trade_activities if a.is_complete(as_of)]
        trade_remaining_acts = [a for a in trade_activities if not a.is_complete(as_of)]

        # Is trade's phase active?
        trade_phases = set(a.phase for a in trade_activities)
        trade_phase_active = phase_id in trade_phases if phase_id else False

        # Expected cost based on schedule
        expected_cost = self.allocator.get_expected_cost_to_date(trade_name, as_of)
        gmp_budget = self.gmp.get(trade_name, 0)
        trade_expected_pct = expected_cost / gmp_budget if gmp_budget > 0 else 0

        # Days to peak / since start
        peak_date = self._get_trade_peak_date(trade_name)
        start_date = self._get_trade_start_date(trade_name)

        days_to_peak = (peak_date - as_of).days if peak_date else 0
        days_since_start = (as_of - start_date).days if start_date else 0

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 3: COST
        # ─────────────────────────────────────────────────────────────────────
        schedule_variance = cumulative_actual_cost - expected_cost
        variance_pct = schedule_variance / expected_cost if expected_cost > 0 else 0

        # Cost velocity (rolling average)
        if cost_history and len(cost_history) > 0:
            # Assume ~30 days per period
            cost_velocity = np.mean(cost_history[-6:]) / 30 if len(cost_history) > 0 else 0
        else:
            cost_velocity = period_cost / 30 if period_cost > 0 else 0

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 4: BUDGET
        # ─────────────────────────────────────────────────────────────────────
        actual_pct_spent = cumulative_actual_cost / gmp_budget if gmp_budget > 0 else 0
        budget_remaining = gmp_budget - cumulative_actual_cost

        # Forecast at completion
        forecast_total, budget_variance = self.allocator.get_forecast_at_completion(
            trade_name, cumulative_actual_cost, as_of
        )

        return ScheduleDrivenFeatures(
            period=period,
            as_of_date=as_of,
            # Level 1 - Project
            project_pct_complete=project_pct,
            project_days_elapsed=days_elapsed,
            project_days_remaining=days_remaining,
            project_total_days=total_days,
            # Level 1 - Phase
            current_phase=phase_id,
            phase_pct_complete=phase_pct,
            phase_days_remaining=max(0, phase_remaining),
            phases_active_count=phases_active,
            # Level 1 - Activity
            total_active_activities=total_active,
            critical_path_activities=critical,
            activities_completed_to_date=completed,
            activities_remaining=remaining,
            avg_float=avg_float,
            min_float=min_float,
            # Level 2 - Trade Schedule
            trade_activities_active=len(trade_active),
            trade_activities_complete=len(trade_complete),
            trade_activities_remaining=len(trade_remaining_acts),
            trade_phase_active=trade_phase_active,
            trade_expected_pct_complete=trade_expected_pct,
            trade_days_to_peak=days_to_peak,
            trade_days_since_start=max(0, days_since_start),
            # Level 3 - Cost
            trade_actual_cost=cumulative_actual_cost,
            trade_cost_this_period=period_cost,
            trade_expected_cost=expected_cost,
            trade_schedule_variance=schedule_variance,
            trade_variance_pct=variance_pct,
            trade_cost_velocity=cost_velocity,
            # Level 4 - Budget
            trade_gmp_budget=gmp_budget,
            trade_budget_remaining=budget_remaining,
            trade_actual_pct_spent=actual_pct_spent,
            trade_forecast_at_completion=forecast_total,
            trade_budget_variance=budget_variance
        )

    def build_training_data(
        self,
        trade_name: str,
        monthly_costs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build complete training dataset for a trade.
        Each row is a month with schedule-driven features.

        Args:
            trade_name: GMP trade name
            monthly_costs: DataFrame with 'year_month' and 'total_cost' columns
        """
        records = []
        cumulative = 0.0
        cost_history = []

        for _, row in monthly_costs.iterrows():
            period = str(row['year_month'])
            monthly_cost = float(row.get('total_cost', 0))
            cumulative += monthly_cost
            cost_history.append(monthly_cost)

            features = self.build_features_for_period(
                trade_name,
                period,
                cumulative,
                monthly_cost,
                cost_history
            )

            # Convert to dict and add target
            record = asdict(features)
            record['monthly_cost'] = monthly_cost
            record['cumulative_cost'] = cumulative

            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Built {len(df)} training samples for {trade_name}")
        return df

    def get_schedule_feature_columns(self) -> List[str]:
        """Get list of schedule-related feature columns"""
        return [
            'project_pct_complete', 'project_days_elapsed', 'project_days_remaining',
            'phase_pct_complete', 'phase_days_remaining', 'phases_active_count',
            'total_active_activities', 'critical_path_activities',
            'activities_completed_to_date', 'activities_remaining',
            'avg_float', 'min_float'
        ]

    def get_trade_schedule_feature_columns(self) -> List[str]:
        """Get list of trade schedule feature columns"""
        return [
            'trade_activities_active', 'trade_activities_complete',
            'trade_activities_remaining', 'trade_phase_active',
            'trade_expected_pct_complete', 'trade_days_to_peak',
            'trade_days_since_start'
        ]

    def get_cost_feature_columns(self) -> List[str]:
        """Get list of cost feature columns"""
        return [
            'trade_actual_cost', 'trade_cost_this_period',
            'trade_expected_cost', 'trade_schedule_variance',
            'trade_variance_pct', 'trade_cost_velocity'
        ]

    def get_budget_feature_columns(self) -> List[str]:
        """Get list of budget feature columns"""
        return [
            'trade_gmp_budget', 'trade_budget_remaining',
            'trade_actual_pct_spent', 'trade_forecast_at_completion',
            'trade_budget_variance'
        ]

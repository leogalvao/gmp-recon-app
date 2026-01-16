"""
Feature Store Service - Computes and stores canonical cost features.

Handles the Phase 2 feature store population:
1. Aggregate costs by canonical trade and period
2. Normalize to per-SF metrics
3. Compute progress metrics
4. Store for ML training
"""
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.models import (
    Project,
    GMP,
    BudgetEntity,
    DirectCostEntity,
    CanonicalTrade,
    CanonicalCostFeature,
    ScheduleActivity,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureBackfillResult:
    """Result of backfilling features for a project."""
    project_id: int
    periods_created: int
    trades_covered: int
    date_range_start: Optional[date]
    date_range_end: Optional[date]
    errors: List[str] = field(default_factory=list)


@dataclass
class PeriodFeatures:
    """Features for a single period and trade."""
    project_id: int
    canonical_trade_id: int
    period_date: date
    period_type: str
    cost_per_sf_cents: int
    cumulative_cost_per_sf_cents: int
    budget_per_sf_cents: int
    pct_complete: Optional[float]
    schedule_pct_elapsed: Optional[float]
    pct_east: float
    pct_west: float
    is_backfill: bool


class FeatureStoreService:
    """
    Service for computing and storing canonical cost features.

    Features are normalized to per-square-foot metrics for cross-project
    comparability. Supports both real-time computation and historical backfill.
    """

    def __init__(self, db: Session):
        self.db = db

    def backfill_project_features(
        self,
        project_id: int,
        period_type: str = 'monthly',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> FeatureBackfillResult:
        """
        Backfill canonical cost features for a project.

        Args:
            project_id: Project to backfill
            period_type: 'weekly' or 'monthly'
            start_date: Start of backfill range (default: project start)
            end_date: End of backfill range (default: today)

        Returns:
            Backfill result with statistics
        """
        project = self.db.query(Project).get(project_id)
        if not project:
            return FeatureBackfillResult(
                project_id=project_id,
                periods_created=0,
                trades_covered=0,
                date_range_start=None,
                date_range_end=None,
                errors=[f"Project {project_id} not found"]
            )

        if not project.total_square_feet or project.total_square_feet <= 0:
            return FeatureBackfillResult(
                project_id=project_id,
                periods_created=0,
                trades_covered=0,
                date_range_start=None,
                date_range_end=None,
                errors=["Project has no square footage set"]
            )

        # Determine date range
        if not start_date:
            start_date = project.start_date or date(2020, 1, 1)
        if not end_date:
            end_date = date.today()

        # Get all GMPs with canonical trades
        gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id != None
        ).all()

        if not gmps:
            return FeatureBackfillResult(
                project_id=project_id,
                periods_created=0,
                trades_covered=0,
                date_range_start=start_date,
                date_range_end=end_date,
                errors=["No GMPs with canonical trade mappings"]
            )

        # Generate periods
        periods = self._generate_periods(start_date, end_date, period_type)

        # Compute features for each trade and period
        features_created = 0
        trades_covered = set()

        for gmp in gmps:
            trade_id = gmp.canonical_trade_id
            trades_covered.add(trade_id)

            # Get budget amount for this GMP
            budget_cents = gmp.original_amount_cents

            # Get cumulative costs up to each period
            cumulative = 0
            for period_start, period_end in periods:
                # Get costs in this period for this GMP's budgets
                period_cost = self._get_period_cost(gmp.id, period_start, period_end)
                cumulative += period_cost

                # Compute per-SF metrics
                cost_per_sf = period_cost // project.total_square_feet if period_cost > 0 else 0
                cumulative_per_sf = cumulative // project.total_square_feet if cumulative > 0 else 0
                budget_per_sf = budget_cents // project.total_square_feet if budget_cents > 0 else 0

                # Get progress metrics
                pct_complete = self._get_schedule_progress(gmp.id, period_end)
                schedule_elapsed = self._get_schedule_elapsed(project, period_end)

                # Get regional splits
                pct_east, pct_west = self._get_regional_splits(gmp)

                # Check if feature already exists
                existing = self.db.query(CanonicalCostFeature).filter(
                    CanonicalCostFeature.project_id == project_id,
                    CanonicalCostFeature.canonical_trade_id == trade_id,
                    CanonicalCostFeature.period_date == period_end,
                    CanonicalCostFeature.period_type == period_type
                ).first()

                if existing:
                    # Update existing
                    existing.cost_per_sf_cents = cost_per_sf
                    existing.cumulative_cost_per_sf_cents = cumulative_per_sf
                    existing.budget_per_sf_cents = budget_per_sf
                    existing.pct_complete = pct_complete
                    existing.schedule_pct_elapsed = schedule_elapsed
                    existing.pct_east = pct_east
                    existing.pct_west = pct_west
                else:
                    # Create new
                    feature = CanonicalCostFeature(
                        project_id=project_id,
                        canonical_trade_id=trade_id,
                        period_date=period_end,
                        period_type=period_type,
                        cost_per_sf_cents=cost_per_sf,
                        cumulative_cost_per_sf_cents=cumulative_per_sf,
                        budget_per_sf_cents=budget_per_sf,
                        pct_complete=pct_complete,
                        schedule_pct_elapsed=schedule_elapsed,
                        pct_east=pct_east,
                        pct_west=pct_west,
                        is_backfill=True
                    )
                    self.db.add(feature)
                    features_created += 1

        self.db.flush()

        return FeatureBackfillResult(
            project_id=project_id,
            periods_created=features_created,
            trades_covered=len(trades_covered),
            date_range_start=start_date,
            date_range_end=end_date
        )

    def backfill_all_projects(
        self,
        period_type: str = 'monthly'
    ) -> Dict[int, FeatureBackfillResult]:
        """Backfill features for all eligible projects."""
        projects = self.db.query(Project).filter(
            Project.is_training_eligible == True,
            Project.total_square_feet != None
        ).all()

        results = {}
        for project in projects:
            try:
                result = self.backfill_project_features(
                    project_id=project.id,
                    period_type=period_type
                )
                results[project.id] = result
                logger.info(
                    f"Backfilled {result.periods_created} features for project {project.code}"
                )
            except Exception as e:
                logger.error(f"Failed to backfill project {project.code}: {e}")
                results[project.id] = FeatureBackfillResult(
                    project_id=project.id,
                    periods_created=0,
                    trades_covered=0,
                    date_range_start=None,
                    date_range_end=None,
                    errors=[str(e)]
                )

        self.db.commit()
        return results

    def compute_current_features(
        self,
        project_id: int,
        as_of_date: Optional[date] = None
    ) -> List[PeriodFeatures]:
        """
        Compute current period features for a project.

        Used for real-time feature computation (not backfill).
        """
        if not as_of_date:
            as_of_date = date.today()

        project = self.db.query(Project).get(project_id)
        if not project or not project.total_square_feet:
            return []

        # Get current month period
        period_start = as_of_date.replace(day=1)
        period_end = as_of_date

        gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id != None
        ).all()

        features = []
        for gmp in gmps:
            # Compute cumulative cost
            cumulative = self._get_cumulative_cost(gmp.id, as_of_date)
            period_cost = self._get_period_cost(gmp.id, period_start, period_end)

            # Per-SF metrics
            cost_per_sf = period_cost // project.total_square_feet if period_cost > 0 else 0
            cumulative_per_sf = cumulative // project.total_square_feet if cumulative > 0 else 0
            budget_per_sf = gmp.original_amount_cents // project.total_square_feet

            # Progress
            pct_complete = self._get_schedule_progress(gmp.id, as_of_date)
            schedule_elapsed = self._get_schedule_elapsed(project, as_of_date)

            # Regional splits
            pct_east, pct_west = self._get_regional_splits(gmp)

            features.append(PeriodFeatures(
                project_id=project_id,
                canonical_trade_id=gmp.canonical_trade_id,
                period_date=as_of_date,
                period_type='monthly',
                cost_per_sf_cents=cost_per_sf,
                cumulative_cost_per_sf_cents=cumulative_per_sf,
                budget_per_sf_cents=budget_per_sf,
                pct_complete=pct_complete,
                schedule_pct_elapsed=schedule_elapsed,
                pct_east=pct_east,
                pct_west=pct_west,
                is_backfill=False
            ))

        return features

    def get_cross_project_features(
        self,
        canonical_trade_id: int,
        period_type: str = 'monthly',
        min_data_quality: float = 0.6
    ) -> List[CanonicalCostFeature]:
        """
        Get features across all projects for a canonical trade.

        Used for training global models.
        """
        return self.db.query(CanonicalCostFeature).join(Project).filter(
            CanonicalCostFeature.canonical_trade_id == canonical_trade_id,
            CanonicalCostFeature.period_type == period_type,
            Project.is_training_eligible == True,
            Project.data_quality_score >= min_data_quality
        ).all()

    def get_feature_statistics(
        self,
        canonical_trade_id: int
    ) -> Dict[str, any]:
        """Get aggregate statistics for a canonical trade's features."""
        features = self.db.query(CanonicalCostFeature).filter(
            CanonicalCostFeature.canonical_trade_id == canonical_trade_id
        ).all()

        if not features:
            return {}

        cost_per_sf_values = [f.cost_per_sf_cents for f in features if f.cost_per_sf_cents > 0]

        return {
            'canonical_trade_id': canonical_trade_id,
            'total_records': len(features),
            'projects_count': len(set(f.project_id for f in features)),
            'avg_cost_per_sf_cents': sum(cost_per_sf_values) // len(cost_per_sf_values) if cost_per_sf_values else 0,
            'min_cost_per_sf_cents': min(cost_per_sf_values) if cost_per_sf_values else 0,
            'max_cost_per_sf_cents': max(cost_per_sf_values) if cost_per_sf_values else 0,
        }

    def _generate_periods(
        self,
        start_date: date,
        end_date: date,
        period_type: str
    ) -> List[Tuple[date, date]]:
        """Generate period boundaries."""
        periods = []
        current = start_date

        if period_type == 'monthly':
            while current <= end_date:
                # Get first day of month
                month_start = current.replace(day=1)
                # Get last day of month
                if current.month == 12:
                    month_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
                else:
                    month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)

                if month_end > end_date:
                    month_end = end_date

                periods.append((month_start, month_end))

                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1, day=1)
                else:
                    current = current.replace(month=current.month + 1, day=1)

        elif period_type == 'weekly':
            # Start from Monday of the week containing start_date
            week_start = start_date - timedelta(days=start_date.weekday())
            while week_start <= end_date:
                week_end = week_start + timedelta(days=6)
                if week_end > end_date:
                    week_end = end_date
                periods.append((week_start, week_end))
                week_start = week_start + timedelta(days=7)

        return periods

    def _get_period_cost(
        self,
        gmp_id: int,
        period_start: date,
        period_end: date
    ) -> int:
        """Get total cost for a GMP in a period."""
        result = self.db.query(func.sum(DirectCostEntity.gross_amount_cents)).join(
            BudgetEntity
        ).filter(
            BudgetEntity.gmp_id == gmp_id,
            DirectCostEntity.transaction_date >= period_start,
            DirectCostEntity.transaction_date <= period_end
        ).scalar()

        return result or 0

    def _get_cumulative_cost(self, gmp_id: int, as_of_date: date) -> int:
        """Get cumulative cost for a GMP up to a date."""
        result = self.db.query(func.sum(DirectCostEntity.gross_amount_cents)).join(
            BudgetEntity
        ).filter(
            BudgetEntity.gmp_id == gmp_id,
            DirectCostEntity.transaction_date <= as_of_date
        ).scalar()

        return result or 0

    def _get_schedule_progress(self, gmp_id: int, as_of_date: date) -> Optional[float]:
        """Get schedule-based progress for a GMP."""
        # Get average progress from linked schedule activities
        gmp = self.db.query(GMP).get(gmp_id)
        if not gmp:
            return None

        # Try to find schedule activities linked to this GMP division
        activities = self.db.query(ScheduleActivity).filter(
            ScheduleActivity.mappings.any(gmp_division=gmp.division)
        ).all()

        if not activities:
            return None

        # Compute weighted average progress
        total_progress = sum(a.progress_pct or 0 for a in activities)
        return total_progress / len(activities) if activities else None

    def _get_schedule_elapsed(self, project: Project, as_of_date: date) -> Optional[float]:
        """Get schedule elapsed percentage for project."""
        if not project.start_date or not project.end_date:
            return None

        total_days = (project.end_date - project.start_date).days
        if total_days <= 0:
            return None

        elapsed_days = (as_of_date - project.start_date).days
        return min(1.0, max(0.0, elapsed_days / total_days))

    def _get_regional_splits(self, gmp: GMP) -> Tuple[float, float]:
        """Get regional split percentages for a GMP."""
        if gmp.zone == 'EAST':
            return 1.0, 0.0
        elif gmp.zone == 'WEST':
            return 0.0, 1.0
        else:  # SHARED
            return 0.5, 0.5

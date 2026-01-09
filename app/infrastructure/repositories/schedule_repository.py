"""
Schedule Repository - Data access layer for Schedule entities.

Implements repository pattern for Schedule operations with:
- Activity management
- GMP linkage
- Progress tracking
- Earned value calculations
"""
from typing import List, Optional, Dict
from datetime import datetime, date
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import ScheduleActivity, ScheduleToGMPMapping
from app.domain.exceptions import (
    ScheduleActivityNotFoundError,
    InvalidScheduleDateRangeError,
)
from .base_repository import BaseRepository


class ScheduleRepository(BaseRepository[ScheduleActivity]):
    """
    Repository for Schedule Activity entities.

    Schedule provides temporal framework for cost forecasting.
    Activities link to GMP divisions for earned value tracking.
    """

    def __init__(self, session: Session):
        super().__init__(session, ScheduleActivity)

    def exists(self, **criteria) -> bool:
        """Check if a ScheduleActivity matching the criteria exists."""
        query = self.session.query(ScheduleActivity)
        for field, value in criteria.items():
            query = query.filter(getattr(ScheduleActivity, field) == value)
        return query.first() is not None

    def get_by_activity_id(self, activity_id: str) -> Optional[ScheduleActivity]:
        """
        Get activity by its activity ID (not database ID).

        Args:
            activity_id: P6/external activity ID

        Returns:
            Schedule activity if found
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.activity_id == activity_id
        ).first()

    def get_by_source_uid(self, source_uid: str) -> Optional[ScheduleActivity]:
        """
        Get activity by its source UID (P6 GUID).

        Args:
            source_uid: P6 GUID

        Returns:
            Schedule activity if found
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.source_uid == source_uid
        ).first()

    def get_by_zone(self, zone: str) -> List[ScheduleActivity]:
        """
        Get all activities for a zone.

        Args:
            zone: Zone identifier (EAST, WEST, SHARED)

        Returns:
            List of activities
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.zone == zone
        ).order_by(ScheduleActivity.start_date).all()

    def get_critical_path(self) -> List[ScheduleActivity]:
        """
        Get all critical path activities.

        Returns:
            List of critical activities (total_float = 0)
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.is_critical == True
        ).order_by(ScheduleActivity.start_date).all()

    def get_in_progress(self) -> List[ScheduleActivity]:
        """
        Get all in-progress activities.

        Returns:
            List of activities currently in progress
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.is_in_progress == True
        ).order_by(ScheduleActivity.start_date).all()

    def get_by_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[ScheduleActivity]:
        """
        Get activities within a date range.

        Args:
            start_date: Range start
            end_date: Range end

        Returns:
            Activities overlapping the range
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.start_date <= end_date,
            ScheduleActivity.finish_date >= start_date
        ).order_by(ScheduleActivity.start_date).all()

    def get_active_on_date(self, target_date: date) -> List[ScheduleActivity]:
        """
        Get activities active on a specific date.

        Args:
            target_date: Date to check

        Returns:
            Activities active on that date
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.start_date <= target_date,
            ScheduleActivity.finish_date >= target_date
        ).all()

    def get_by_gmp_division(self, gmp_division: str) -> List[ScheduleActivity]:
        """
        Get all activities linked to a GMP division.

        Args:
            gmp_division: GMP division name

        Returns:
            List of linked activities
        """
        return self.session.query(ScheduleActivity).join(
            ScheduleToGMPMapping,
            ScheduleActivity.id == ScheduleToGMPMapping.schedule_activity_id
        ).filter(
            ScheduleToGMPMapping.gmp_division == gmp_division
        ).order_by(ScheduleActivity.start_date).all()

    def update_progress(
        self,
        activity_id: int,
        percent_complete: int
    ) -> ScheduleActivity:
        """
        Update activity percent complete.

        Args:
            activity_id: Database ID
            percent_complete: New percent complete (0-100)

        Returns:
            Updated activity

        Raises:
            ScheduleActivityNotFoundError: If not found
        """
        activity = self.get_by_id(activity_id)
        if not activity:
            raise ScheduleActivityNotFoundError(str(activity_id))

        activity.pct_complete = min(100, max(0, percent_complete))
        activity.progress_pct = activity.pct_complete / 100.0

        # Update state flags
        if activity.pct_complete == 100:
            activity.is_complete = True
            activity.is_in_progress = False
        elif activity.pct_complete > 0:
            activity.is_complete = False
            activity.is_in_progress = True
        else:
            activity.is_complete = False
            activity.is_in_progress = False

        return activity

    def update_zone(self, activity_id: int, zone: str) -> ScheduleActivity:
        """
        Update activity zone assignment.

        Args:
            activity_id: Database ID
            zone: New zone (EAST, WEST, SHARED)

        Returns:
            Updated activity
        """
        activity = self.get_by_id(activity_id)
        if not activity:
            raise ScheduleActivityNotFoundError(str(activity_id))

        activity.zone = zone
        return activity

    # GMP Mapping operations

    def get_gmp_mappings(self, activity_id: int) -> List[ScheduleToGMPMapping]:
        """
        Get all GMP mappings for an activity.

        Args:
            activity_id: Activity database ID

        Returns:
            List of GMP mappings
        """
        return self.session.query(ScheduleToGMPMapping).filter(
            ScheduleToGMPMapping.schedule_activity_id == activity_id
        ).all()

    def add_gmp_mapping(
        self,
        activity_id: int,
        gmp_division: str,
        weight: float = 1.0
    ) -> ScheduleToGMPMapping:
        """
        Add a GMP mapping for an activity.

        Args:
            activity_id: Activity database ID
            gmp_division: GMP division name
            weight: Allocation weight (0.0-1.0)

        Returns:
            Created mapping
        """
        mapping = ScheduleToGMPMapping(
            schedule_activity_id=activity_id,
            gmp_division=gmp_division,
            weight=weight,
            created_at=datetime.utcnow()
        )
        self.session.add(mapping)
        return mapping

    def remove_gmp_mapping(self, activity_id: int, gmp_division: str) -> bool:
        """
        Remove a GMP mapping.

        Args:
            activity_id: Activity database ID
            gmp_division: GMP division name

        Returns:
            True if mapping was found and removed
        """
        mapping = self.session.query(ScheduleToGMPMapping).filter(
            ScheduleToGMPMapping.schedule_activity_id == activity_id,
            ScheduleToGMPMapping.gmp_division == gmp_division
        ).first()

        if mapping:
            self.session.delete(mapping)
            return True
        return False

    # Aggregation and calculations

    def get_schedule_summary(self) -> Dict:
        """
        Get overall schedule summary statistics.

        Returns:
            Dict with schedule metrics
        """
        activities = self.get_all()

        if not activities:
            return {
                'total_activities': 0,
                'critical_count': 0,
                'complete_count': 0,
                'in_progress_count': 0,
                'not_started_count': 0,
                'overall_progress_pct': 0,
                'earliest_start': None,
                'latest_finish': None
            }

        complete = sum(1 for a in activities if a.is_complete)
        in_progress = sum(1 for a in activities if a.is_in_progress)
        critical = sum(1 for a in activities if a.is_critical)

        start_dates = [a.start_date for a in activities if a.start_date]
        finish_dates = [a.finish_date for a in activities if a.finish_date]

        # Calculate overall progress as weighted average
        total_duration = sum(a.duration_days or 0 for a in activities)
        weighted_progress = sum(
            (a.progress_pct or 0) * (a.duration_days or 0)
            for a in activities
        )
        overall_progress = (weighted_progress / total_duration * 100) if total_duration > 0 else 0

        return {
            'total_activities': len(activities),
            'critical_count': critical,
            'complete_count': complete,
            'in_progress_count': in_progress,
            'not_started_count': len(activities) - complete - in_progress,
            'overall_progress_pct': round(overall_progress, 2),
            'earliest_start': min(start_dates).isoformat() if start_dates else None,
            'latest_finish': max(finish_dates).isoformat() if finish_dates else None
        }

    def get_gmp_division_progress(self, gmp_division: str) -> Dict:
        """
        Get progress for a specific GMP division.

        Aggregates progress across all linked activities.

        Args:
            gmp_division: GMP division name

        Returns:
            Dict with progress metrics
        """
        activities = self.get_by_gmp_division(gmp_division)

        if not activities:
            return {
                'gmp_division': gmp_division,
                'activity_count': 0,
                'progress_pct': 0,
                'complete_count': 0,
                'in_progress_count': 0
            }

        total_duration = sum(a.duration_days or 1 for a in activities)
        weighted_progress = sum(
            (a.progress_pct or 0) * (a.duration_days or 1)
            for a in activities
        )
        overall_progress = (weighted_progress / total_duration * 100) if total_duration > 0 else 0

        return {
            'gmp_division': gmp_division,
            'activity_count': len(activities),
            'progress_pct': round(overall_progress, 2),
            'complete_count': sum(1 for a in activities if a.is_complete),
            'in_progress_count': sum(1 for a in activities if a.is_in_progress)
        }

    def calculate_earned_value(
        self,
        gmp_division: str,
        budget_amount_cents: int,
        as_of_date: Optional[date] = None
    ) -> Dict:
        """
        Calculate earned value metrics for a GMP division.

        Implements earned value formulas from the spec:
        - PV (Planned Value): Budget × expected % complete
        - EV (Earned Value): Budget × actual % complete

        Args:
            gmp_division: GMP division name
            budget_amount_cents: BAC (Budget at Completion) in cents
            as_of_date: Reference date (default: today)

        Returns:
            Dict with PV, EV, and schedule metrics
        """
        if as_of_date is None:
            as_of_date = date.today()

        activities = self.get_by_gmp_division(gmp_division)

        if not activities:
            return {
                'gmp_division': gmp_division,
                'bac_cents': budget_amount_cents,
                'pv_cents': 0,
                'ev_cents': 0,
                'expected_pct_complete': 0,
                'actual_pct_complete': 0,
                'schedule_variance_cents': 0,
                'spi': 0
            }

        # Calculate expected % complete based on schedule
        total_duration = sum(a.duration_days or 1 for a in activities)
        expected_weighted = 0
        actual_weighted = 0

        for activity in activities:
            duration = activity.duration_days or 1
            weight = duration / total_duration

            # Expected progress based on elapsed time
            if activity.start_date and activity.finish_date:
                if as_of_date >= activity.finish_date:
                    expected_pct = 1.0
                elif as_of_date <= activity.start_date:
                    expected_pct = 0.0
                else:
                    elapsed = (as_of_date - activity.start_date).days
                    total = (activity.finish_date - activity.start_date).days
                    expected_pct = elapsed / total if total > 0 else 0
            else:
                expected_pct = 0

            expected_weighted += expected_pct * weight
            actual_weighted += (activity.progress_pct or 0) * weight

        # Calculate PV and EV
        pv_cents = int(budget_amount_cents * expected_weighted)
        ev_cents = int(budget_amount_cents * actual_weighted)

        # Schedule variance and SPI
        sv_cents = ev_cents - pv_cents
        spi = ev_cents / pv_cents if pv_cents > 0 else 0

        return {
            'gmp_division': gmp_division,
            'bac_cents': budget_amount_cents,
            'pv_cents': pv_cents,
            'ev_cents': ev_cents,
            'expected_pct_complete': round(expected_weighted * 100, 2),
            'actual_pct_complete': round(actual_weighted * 100, 2),
            'schedule_variance_cents': sv_cents,
            'spi': round(spi, 3)
        }

    def get_unassigned_to_zone(self) -> List[ScheduleActivity]:
        """
        Get activities without zone assignment.

        Returns:
            List of activities with null zone
        """
        return self.session.query(ScheduleActivity).filter(
            ScheduleActivity.zone.is_(None)
        ).all()

    def bulk_update_zone(self, activity_ids: List[int], zone: str) -> int:
        """
        Bulk update zone for multiple activities.

        Args:
            activity_ids: List of activity IDs
            zone: Zone to assign

        Returns:
            Count of updated activities
        """
        count = self.session.query(ScheduleActivity).filter(
            ScheduleActivity.id.in_(activity_ids)
        ).update({'zone': zone}, synchronize_session=False)
        return count

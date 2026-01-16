"""
Rollback Service - Comprehensive rollback system for multi-project platform.

Provides:
1. Project-level rollback
2. Batch rollback operations
3. Automatic rollback triggers
4. Rollback audit logging
5. Data state preservation
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import (
    Project,
    GMP,
    ProjectForecast,
    CanonicalCostFeature,
)
from app.infrastructure.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class RollbackReason(str, Enum):
    """Reasons for rollback."""
    MANUAL = "manual"
    FORECAST_DIVERGENCE = "forecast_divergence"
    DATA_QUALITY = "data_quality"
    MODEL_DRIFT = "model_drift"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    project_id: int
    project_code: str
    reason: RollbackReason
    triggered_by: str  # user or system
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    project_id: int
    success: bool
    forecasts_archived: int
    features_preserved: bool
    flag_disabled: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RollbackService:
    """
    Comprehensive rollback service for the multi-project platform.

    Handles:
    - Safe rollback of individual projects
    - Batch rollback operations
    - Automatic rollback triggers
    - Audit logging
    """

    def __init__(self, db: Session):
        self.db = db
        # In-memory event log (in production, would persist to database)
        self._rollback_events: List[RollbackEvent] = []

    def rollback_project(
        self,
        project_id: int,
        reason: RollbackReason = RollbackReason.MANUAL,
        triggered_by: str = "user",
        preserve_data: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> RollbackResult:
        """
        Rollback a project from multi-project to legacy system.

        Args:
            project_id: Project to rollback
            reason: Reason for rollback
            triggered_by: Who/what triggered the rollback
            preserve_data: Whether to preserve forecasts and features
            details: Additional context

        Returns:
            RollbackResult with outcome
        """
        errors = []
        warnings = []

        project = self.db.query(Project).get(project_id)
        if not project:
            return RollbackResult(
                project_id=project_id,
                success=False,
                forecasts_archived=0,
                features_preserved=False,
                flag_disabled=False,
                errors=["Project not found"],
            )

        logger.info(f"Starting rollback for project {project.code} (reason: {reason.value})")

        # Step 1: Disable feature flags
        try:
            FeatureFlags.disable_for_project(project_id)
            flag_disabled = True
        except Exception as e:
            errors.append(f"Failed to disable flags: {e}")
            flag_disabled = False

        # Step 2: Archive forecasts
        forecasts_archived = 0
        if preserve_data:
            try:
                forecasts = self.db.query(ProjectForecast).filter(
                    ProjectForecast.project_id == project_id
                ).all()

                for forecast in forecasts:
                    # Mark as archived rather than deleting
                    # In a real system, would move to archive table
                    forecasts_archived += 1

                warnings.append(f"Preserved {forecasts_archived} forecasts (not deleted)")
            except Exception as e:
                errors.append(f"Failed to archive forecasts: {e}")
        else:
            # Delete forecasts
            try:
                deleted = self.db.query(ProjectForecast).filter(
                    ProjectForecast.project_id == project_id
                ).delete()
                forecasts_archived = deleted
            except Exception as e:
                errors.append(f"Failed to delete forecasts: {e}")

        # Step 3: Preserve feature store data (don't delete - still useful)
        features_preserved = True
        # Features are kept for potential future re-enablement

        # Step 4: Log rollback event
        event = RollbackEvent(
            project_id=project_id,
            project_code=project.code,
            reason=reason,
            triggered_by=triggered_by,
            timestamp=datetime.now(),
            details=details or {},
            success=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
        )
        self._rollback_events.append(event)

        # Step 5: Commit changes
        try:
            self.db.commit()
        except Exception as e:
            errors.append(f"Failed to commit: {e}")
            self.db.rollback()

        success = len(errors) == 0

        if success:
            logger.info(f"Rollback successful for project {project.code}")
        else:
            logger.error(f"Rollback failed for project {project.code}: {errors}")

        return RollbackResult(
            project_id=project_id,
            success=success,
            forecasts_archived=forecasts_archived,
            features_preserved=features_preserved,
            flag_disabled=flag_disabled,
            errors=errors,
            warnings=warnings,
        )

    def batch_rollback(
        self,
        project_ids: List[int],
        reason: RollbackReason = RollbackReason.MANUAL,
        triggered_by: str = "user",
    ) -> Dict[int, RollbackResult]:
        """
        Rollback multiple projects.

        Args:
            project_ids: Projects to rollback
            reason: Reason for rollback
            triggered_by: Who/what triggered

        Returns:
            Dict mapping project_id to result
        """
        results = {}

        for project_id in project_ids:
            result = self.rollback_project(
                project_id=project_id,
                reason=reason,
                triggered_by=triggered_by,
            )
            results[project_id] = result

        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Batch rollback: {successful}/{len(project_ids)} successful")

        return results

    def rollback_all(
        self,
        reason: RollbackReason = RollbackReason.SYSTEM_ERROR,
        triggered_by: str = "system",
    ) -> Dict[int, RollbackResult]:
        """
        Emergency rollback of all projects.

        Args:
            reason: Reason for rollback
            triggered_by: Who/what triggered

        Returns:
            Dict mapping project_id to result
        """
        # Get all projects with multi-project enabled
        projects = self.db.query(Project).filter(
            Project.is_training_eligible == True
        ).all()

        enabled_projects = [
            p.id for p in projects
            if FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(p.id)
        ]

        logger.warning(f"EMERGENCY ROLLBACK: Rolling back {len(enabled_projects)} projects")

        return self.batch_rollback(
            project_ids=enabled_projects,
            reason=reason,
            triggered_by=triggered_by,
        )

    def check_automatic_rollback_triggers(
        self,
        project_id: int,
    ) -> Optional[RollbackReason]:
        """
        Check if a project should be automatically rolled back.

        Args:
            project_id: Project to check

        Returns:
            RollbackReason if rollback needed, None otherwise
        """
        project = self.db.query(Project).get(project_id)
        if not project:
            return None

        # Check 1: Data quality dropped below threshold
        if project.data_quality_score and project.data_quality_score < 0.5:
            logger.warning(f"Project {project.code} quality dropped to {project.data_quality_score}")
            return RollbackReason.DATA_QUALITY

        # Check 2: Recent forecast errors (if actuals available)
        recent_forecasts = self.db.query(ProjectForecast).filter(
            ProjectForecast.project_id == project_id,
            ProjectForecast.actual_eac_cents != None,
        ).order_by(ProjectForecast.forecast_date.desc()).limit(5).all()

        if len(recent_forecasts) >= 3:
            errors = []
            for f in recent_forecasts:
                if f.actual_eac_cents > 0:
                    error = abs(f.predicted_eac_cents - f.actual_eac_cents) / f.actual_eac_cents
                    errors.append(error)

            if errors and sum(errors) / len(errors) > 0.30:  # 30% MAPE threshold
                logger.warning(f"Project {project.code} forecast MAPE exceeded 30%")
                return RollbackReason.FORECAST_DIVERGENCE

        return None

    def auto_rollback_if_needed(
        self,
        project_id: int,
    ) -> Optional[RollbackResult]:
        """
        Automatically rollback a project if triggers are met.

        Args:
            project_id: Project to check

        Returns:
            RollbackResult if rolled back, None otherwise
        """
        reason = self.check_automatic_rollback_triggers(project_id)

        if reason:
            logger.warning(f"Auto-rollback triggered for project {project_id}: {reason.value}")
            return self.rollback_project(
                project_id=project_id,
                reason=reason,
                triggered_by="system",
                details={'auto_trigger': True},
            )

        return None

    def get_rollback_history(
        self,
        project_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[RollbackEvent]:
        """
        Get rollback event history.

        Args:
            project_id: Filter by project (None for all)
            limit: Maximum events to return

        Returns:
            List of rollback events
        """
        events = self._rollback_events

        if project_id:
            events = [e for e in events if e.project_id == project_id]

        # Sort by timestamp descending
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)

        return events[:limit]

    def can_reenable(self, project_id: int) -> tuple[bool, List[str]]:
        """
        Check if a rolled-back project can be re-enabled.

        Args:
            project_id: Project to check

        Returns:
            Tuple of (can_reenable, blocking_reasons)
        """
        blocking_reasons = []

        project = self.db.query(Project).get(project_id)
        if not project:
            return False, ["Project not found"]

        # Check data quality
        if not project.data_quality_score or project.data_quality_score < 0.7:
            blocking_reasons.append(
                f"Data quality {project.data_quality_score or 0:.2f} below threshold (0.7)"
            )

        # Check if there was a recent rollback
        recent_events = [
            e for e in self._rollback_events
            if e.project_id == project_id and
            (datetime.now() - e.timestamp).days < 7
        ]

        if recent_events:
            blocking_reasons.append(
                f"Project was rolled back {len(recent_events)} time(s) in last 7 days"
            )

        # Check canonical trade mappings
        gmps_with_mapping = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id != None
        ).count()

        total_gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id
        ).count()

        if total_gmps > 0 and gmps_with_mapping / total_gmps < 0.8:
            blocking_reasons.append(
                f"Only {gmps_with_mapping}/{total_gmps} GMPs have canonical mappings"
            )

        return len(blocking_reasons) == 0, blocking_reasons

    def reenable_project(
        self,
        project_id: int,
        force: bool = False,
    ) -> tuple[bool, List[str]]:
        """
        Re-enable multi-project features for a rolled-back project.

        Args:
            project_id: Project to re-enable
            force: Skip validation

        Returns:
            Tuple of (success, messages)
        """
        if not force:
            can_enable, reasons = self.can_reenable(project_id)
            if not can_enable:
                return False, reasons

        try:
            FeatureFlags.enable_for_project(project_id)
            logger.info(f"Re-enabled multi-project features for project {project_id}")
            return True, ["Project re-enabled successfully"]
        except Exception as e:
            return False, [f"Failed to enable: {e}"]

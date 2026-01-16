"""
Backward Compatibility Layer - Ensures smooth transition from legacy to multi-project system.

Provides:
1. Legacy API format compatibility
2. Data format conversion between old and new schemas
3. Gradual cutover support with validation
"""
import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.models import (
    Project,
    GMP,
    CanonicalTrade,
    ProjectForecast,
    ForecastSnapshot,
)
from app.infrastructure.feature_flags import FeatureFlags
from app.domain.services import ForecastInferenceService, ProjectForecastResult

logger = logging.getLogger(__name__)


@dataclass
class CutoverResult:
    """Result of a project cutover operation."""
    project_id: int
    project_code: str
    success: bool
    cutover_time: datetime
    quality_score: float
    forecast_divergence: float
    errors: List[str]
    warnings: List[str]


@dataclass
class ForecastComparison:
    """Comparison between legacy and new forecast systems."""
    project_id: int
    trade_count: int
    max_divergence: float
    mean_divergence: float
    divergent_trades: List[Dict[str, Any]]


class CompatibilityLayer:
    """
    Ensures backward compatibility during migration from legacy to multi-project system.

    Key responsibilities:
    - Convert between legacy and new data formats
    - Support gradual cutover with validation
    - Maintain API contract compatibility
    """

    def __init__(self, db: Session):
        self.db = db

    def get_gmp_with_canonical_info(self, gmp_id: int) -> Dict[str, Any]:
        """
        Return GMP in both legacy and new format.

        Adds canonical trade information while preserving legacy fields.

        Args:
            gmp_id: GMP ID

        Returns:
            Dict with both legacy and new fields
        """
        gmp = self.db.query(GMP).get(gmp_id)
        if not gmp:
            return None

        response = {
            # Legacy fields (unchanged)
            'id': gmp.id,
            'division': gmp.division,
            'zone': gmp.zone,
            'original_amount_cents': gmp.original_amount_cents,
            'current_amount_cents': gmp.current_amount_cents,
            'committed_cost_cents': gmp.committed_cost_cents,
            'project_id': gmp.project_id,

            # New fields (additive)
            'canonical_trade_id': gmp.canonical_trade_id,
            'canonical_trade_code': None,
            'canonical_trade_name': None,
            'normalized_amount_per_sf_cents': gmp.normalized_amount_per_sf_cents,
        }

        # Add canonical trade info if available
        if gmp.canonical_trade_id:
            trade = self.db.query(CanonicalTrade).get(gmp.canonical_trade_id)
            if trade:
                response['canonical_trade_code'] = trade.canonical_code
                response['canonical_trade_name'] = trade.canonical_name

        return response

    def get_forecast_legacy_format(
        self,
        project_id: int,
        use_new_system: bool = None,
    ) -> List[Dict[str, Any]]:
        """
        Get forecasts in legacy ForecastSnapshot format.

        Automatically decides whether to use new or legacy system based on
        feature flags, or can be forced to use specific system.

        Args:
            project_id: Project ID
            use_new_system: Force use of new (True) or legacy (False) system

        Returns:
            List of forecasts in legacy format
        """
        # Determine which system to use
        if use_new_system is None:
            use_new_system = FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(project_id)

        if use_new_system:
            return self._get_forecast_from_new_system(project_id)
        else:
            return self._get_forecast_from_legacy_system(project_id)

    def _get_forecast_from_new_system(self, project_id: int) -> List[Dict[str, Any]]:
        """Get forecasts from new multi-project system and convert to legacy format."""
        service = ForecastInferenceService(self.db)
        forecast = service.get_project_forecast(project_id)

        if not forecast:
            return []

        legacy_forecasts = []
        for tf in forecast.trade_forecasts:
            # Get original GMP division name
            gmp = self.db.query(GMP).filter(
                GMP.project_id == project_id,
                GMP.canonical_trade_id == tf.canonical_trade_id
            ).first()

            legacy_forecasts.append({
                'gmp_division': gmp.division if gmp else tf.canonical_name,
                'eac_cents': int(tf.forecasted_eac * 100),
                'eac_east_cents': int(tf.forecasted_eac * 100 * 0.5),  # Simplified split
                'eac_west_cents': int(tf.forecasted_eac * 100 * 0.5),
                'etc_cents': int((tf.forecasted_eac - tf.current_cumulative_cost) * 100),
                'var_cents': int((tf.budget - tf.forecasted_eac) * 100),
                'confidence_band': self._score_to_band(tf.forecast.std / tf.forecast.mean if tf.forecast.mean else 1),
                # New fields for gradual migration
                '_canonical_trade_id': tf.canonical_trade_id,
                '_canonical_code': tf.canonical_code,
                '_confidence_level': tf.forecast.confidence_level,
                '_lower_bound_cents': int(tf.forecast.lower_bound * 100),
                '_upper_bound_cents': int(tf.forecast.upper_bound * 100),
            })

        return legacy_forecasts

    def _get_forecast_from_legacy_system(self, project_id: int) -> List[Dict[str, Any]]:
        """Get forecasts from legacy ForecastSnapshot system."""
        # Get latest forecast snapshot
        snapshot = self.db.query(ForecastSnapshot).filter(
            ForecastSnapshot.project_id == project_id
        ).order_by(ForecastSnapshot.snapshot_date.desc()).first()

        if not snapshot:
            return []

        # Convert snapshot data to list format
        # (Assuming ForecastSnapshot stores per-division data)
        return snapshot.forecast_data if hasattr(snapshot, 'forecast_data') else []

    def _score_to_band(self, cv: float) -> str:
        """Convert coefficient of variation to confidence band."""
        if cv < 0.05:
            return 'HIGH'
        elif cv < 0.15:
            return 'MEDIUM'
        else:
            return 'LOW'

    def compare_forecasts(
        self,
        project_id: int,
    ) -> ForecastComparison:
        """
        Compare forecasts from legacy and new systems.

        Used for validation before cutover.

        Args:
            project_id: Project to compare

        Returns:
            ForecastComparison with divergence metrics
        """
        legacy_forecasts = self._get_forecast_from_legacy_system(project_id)
        new_forecasts = self._get_forecast_from_new_system(project_id)

        if not legacy_forecasts or not new_forecasts:
            return ForecastComparison(
                project_id=project_id,
                trade_count=0,
                max_divergence=0.0,
                mean_divergence=0.0,
                divergent_trades=[],
            )

        # Match forecasts by division name
        legacy_by_div = {f['gmp_division']: f for f in legacy_forecasts}
        new_by_div = {f['gmp_division']: f for f in new_forecasts}

        divergences = []
        divergent_trades = []

        for div_name, legacy in legacy_by_div.items():
            new = new_by_div.get(div_name)
            if not new:
                continue

            legacy_eac = legacy.get('eac_cents', 0)
            new_eac = new.get('eac_cents', 0)

            if legacy_eac > 0:
                divergence = abs(new_eac - legacy_eac) / legacy_eac
            else:
                divergence = 1.0 if new_eac > 0 else 0.0

            divergences.append(divergence)

            if divergence > 0.1:  # 10% threshold
                divergent_trades.append({
                    'division': div_name,
                    'legacy_eac_cents': legacy_eac,
                    'new_eac_cents': new_eac,
                    'divergence': divergence,
                })

        return ForecastComparison(
            project_id=project_id,
            trade_count=len(divergences),
            max_divergence=max(divergences) if divergences else 0.0,
            mean_divergence=sum(divergences) / len(divergences) if divergences else 0.0,
            divergent_trades=divergent_trades,
        )


class ProjectCutoverService:
    """
    Service for managing project cutover from legacy to multi-project system.

    Ensures safe transition with validation checkpoints.
    """

    def __init__(self, db: Session):
        self.db = db
        self.compatibility = CompatibilityLayer(db)

    def validate_cutover_readiness(
        self,
        project_id: int,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a project is ready for cutover.

        Checks:
        1. Data quality score >= 0.8
        2. Canonical trade mappings exist
        3. Feature store data exists
        4. Forecast divergence <= 20%

        Args:
            project_id: Project to validate

        Returns:
            Tuple of (is_ready, list_of_issues)
        """
        issues = []

        # 1. Check project exists and has quality score
        project = self.db.query(Project).get(project_id)
        if not project:
            return False, ["Project not found"]

        if not project.data_quality_score or project.data_quality_score < 0.8:
            issues.append(
                f"Data quality score {project.data_quality_score or 0:.2f} < 0.8"
            )

        # 2. Check canonical trade mappings
        gmps_with_mapping = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id != None
        ).count()

        total_gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id
        ).count()

        if total_gmps > 0 and gmps_with_mapping / total_gmps < 0.9:
            issues.append(
                f"Only {gmps_with_mapping}/{total_gmps} GMPs have canonical mappings"
            )

        # 3. Check feature store data
        from app.models import CanonicalCostFeature
        feature_count = self.db.query(CanonicalCostFeature).filter(
            CanonicalCostFeature.project_id == project_id
        ).count()

        if feature_count < 12:  # At least 12 months of features
            issues.append(
                f"Only {feature_count} feature records (need at least 12)"
            )

        # 4. Compare forecasts
        comparison = self.compatibility.compare_forecasts(project_id)
        if comparison.max_divergence > 0.2:
            issues.append(
                f"Forecast divergence {comparison.max_divergence:.1%} > 20%"
            )
            for trade in comparison.divergent_trades[:3]:
                issues.append(f"  - {trade['division']}: {trade['divergence']:.1%} divergence")

        return len(issues) == 0, issues

    def execute_cutover(
        self,
        project_id: int,
        force: bool = False,
    ) -> CutoverResult:
        """
        Execute cutover for a project from legacy to multi-project system.

        Args:
            project_id: Project to cut over
            force: Skip validation checks (dangerous)

        Returns:
            CutoverResult with outcome
        """
        errors = []
        warnings = []

        # Get project
        project = self.db.query(Project).get(project_id)
        if not project:
            return CutoverResult(
                project_id=project_id,
                project_code="UNKNOWN",
                success=False,
                cutover_time=datetime.now(),
                quality_score=0.0,
                forecast_divergence=0.0,
                errors=["Project not found"],
                warnings=[],
            )

        # Validate readiness
        if not force:
            is_ready, issues = self.validate_cutover_readiness(project_id)
            if not is_ready:
                return CutoverResult(
                    project_id=project_id,
                    project_code=project.code,
                    success=False,
                    cutover_time=datetime.now(),
                    quality_score=project.data_quality_score or 0.0,
                    forecast_divergence=0.0,
                    errors=issues,
                    warnings=[],
                )

        # Get metrics for result
        comparison = self.compatibility.compare_forecasts(project_id)

        # Enable feature flags for project
        try:
            FeatureFlags.enable_for_project(project_id)
            logger.info(f"Cutover complete for project {project.code}")

        except Exception as e:
            errors.append(f"Failed to enable feature flags: {e}")
            return CutoverResult(
                project_id=project_id,
                project_code=project.code,
                success=False,
                cutover_time=datetime.now(),
                quality_score=project.data_quality_score or 0.0,
                forecast_divergence=comparison.max_divergence,
                errors=errors,
                warnings=warnings,
            )

        return CutoverResult(
            project_id=project_id,
            project_code=project.code,
            success=True,
            cutover_time=datetime.now(),
            quality_score=project.data_quality_score or 0.0,
            forecast_divergence=comparison.max_divergence,
            errors=errors,
            warnings=warnings,
        )

    def rollback_cutover(self, project_id: int) -> bool:
        """
        Rollback a project from multi-project to legacy system.

        Args:
            project_id: Project to rollback

        Returns:
            True if rollback successful
        """
        try:
            FeatureFlags.disable_for_project(project_id)
            logger.info(f"Rollback complete for project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed for project {project_id}: {e}")
            return False

    def get_cutover_status(self, project_id: int) -> Dict[str, Any]:
        """
        Get current cutover status for a project.

        Args:
            project_id: Project to check

        Returns:
            Status dictionary
        """
        project = self.db.query(Project).get(project_id)
        if not project:
            return {'error': 'Project not found'}

        is_ready, issues = self.validate_cutover_readiness(project_id)

        return {
            'project_id': project_id,
            'project_code': project.code,
            'is_multi_project_enabled': FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(project_id),
            'is_ready_for_cutover': is_ready,
            'blocking_issues': issues,
            'data_quality_score': project.data_quality_score,
        }

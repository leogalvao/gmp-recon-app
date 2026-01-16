"""
Monitoring Service - System health, drift detection, and alerting.

Provides:
1. Model performance monitoring
2. Data drift detection
3. System health checks
4. Automated alerting
5. Retraining triggers
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import (
    Project,
    ProjectForecast,
    CanonicalCostFeature,
    MLModelRegistry,
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    MODEL_DRIFT = "model_drift"
    DATA_DRIFT = "data_drift"
    PREDICTION_ERROR = "prediction_error"
    SYSTEM_ERROR = "system_error"
    RETRAINING_NEEDED = "retraining_needed"
    CUTOVER_ISSUE = "cutover_issue"
    QUALITY_DEGRADATION = "quality_degradation"


@dataclass
class Alert:
    """An alert to be sent to operators."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    project_id: Optional[int] = None
    model_id: Optional[int] = None


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report on data or model drift."""
    drift_detected: bool
    drift_score: float
    affected_features: List[str]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


class MonitoringService:
    """
    Service for monitoring ML system health and performance.

    Provides:
    - Model performance tracking
    - Data drift detection
    - Automated alerting
    - Retraining triggers
    """

    # Thresholds for alerts
    MAPE_WARNING_THRESHOLD = 0.15  # 15%
    MAPE_ERROR_THRESHOLD = 0.25   # 25%
    DRIFT_WARNING_THRESHOLD = 0.10
    DRIFT_ERROR_THRESHOLD = 0.20
    MIN_PREDICTIONS_FOR_ANALYSIS = 10

    def __init__(self, db: Session, alert_handlers: Optional[List[Callable[[Alert], None]]] = None):
        self.db = db
        self.alert_handlers = alert_handlers or [self._default_alert_handler]

    def run_health_checks(self) -> List[HealthCheckResult]:
        """
        Run all system health checks.

        Returns:
            List of health check results
        """
        results = []

        # Check 1: Model availability
        results.append(self._check_model_availability())

        # Check 2: Recent predictions
        results.append(self._check_recent_predictions())

        # Check 3: Feature store freshness
        results.append(self._check_feature_freshness())

        # Check 4: Data quality
        results.append(self._check_data_quality())

        # Check 5: Prediction accuracy
        results.append(self._check_prediction_accuracy())

        return results

    def _check_model_availability(self) -> HealthCheckResult:
        """Check if active models are available."""
        active_model = self.db.query(MLModelRegistry).filter(
            MLModelRegistry.is_production == True,
            MLModelRegistry.model_type == 'multi_project_forecaster',
        ).first()

        if active_model:
            return HealthCheckResult(
                check_name="model_availability",
                status="healthy",
                message=f"Active model: {active_model.model_name} v{active_model.model_version}",
                metrics={'model_id': active_model.id, 'version': active_model.model_version},
            )
        else:
            return HealthCheckResult(
                check_name="model_availability",
                status="unhealthy",
                message="No active model found",
                metrics={},
            )

    def _check_recent_predictions(self) -> HealthCheckResult:
        """Check if predictions are being generated."""
        recent_cutoff = date.today() - timedelta(days=7)

        recent_count = self.db.query(ProjectForecast).filter(
            ProjectForecast.forecast_date >= recent_cutoff
        ).count()

        if recent_count > 0:
            return HealthCheckResult(
                check_name="recent_predictions",
                status="healthy",
                message=f"{recent_count} predictions in last 7 days",
                metrics={'prediction_count': recent_count},
            )
        else:
            return HealthCheckResult(
                check_name="recent_predictions",
                status="degraded",
                message="No predictions in last 7 days",
                metrics={'prediction_count': 0},
            )

    def _check_feature_freshness(self) -> HealthCheckResult:
        """Check if feature store is being updated."""
        latest_feature = self.db.query(
            func.max(CanonicalCostFeature.period_date)
        ).scalar()

        if latest_feature:
            days_old = (date.today() - latest_feature).days

            if days_old <= 30:
                status = "healthy"
            elif days_old <= 60:
                status = "degraded"
            else:
                status = "unhealthy"

            return HealthCheckResult(
                check_name="feature_freshness",
                status=status,
                message=f"Latest features from {latest_feature} ({days_old} days old)",
                metrics={'latest_date': str(latest_feature), 'days_old': days_old},
            )
        else:
            return HealthCheckResult(
                check_name="feature_freshness",
                status="unhealthy",
                message="No features in store",
                metrics={},
            )

    def _check_data_quality(self) -> HealthCheckResult:
        """Check overall data quality across projects."""
        avg_quality = self.db.query(
            func.avg(Project.data_quality_score)
        ).filter(
            Project.is_training_eligible == True
        ).scalar()

        if avg_quality is None:
            return HealthCheckResult(
                check_name="data_quality",
                status="degraded",
                message="No quality scores available",
                metrics={},
            )

        if avg_quality >= 0.7:
            status = "healthy"
        elif avg_quality >= 0.5:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthCheckResult(
            check_name="data_quality",
            status=status,
            message=f"Average data quality: {avg_quality:.2f}",
            metrics={'avg_quality': float(avg_quality)},
        )

    def _check_prediction_accuracy(self) -> HealthCheckResult:
        """Check prediction accuracy where actuals are available."""
        # Get forecasts with actuals
        forecasts_with_actuals = self.db.query(ProjectForecast).filter(
            ProjectForecast.actual_eac_cents != None
        ).all()

        if len(forecasts_with_actuals) < self.MIN_PREDICTIONS_FOR_ANALYSIS:
            return HealthCheckResult(
                check_name="prediction_accuracy",
                status="degraded",
                message=f"Insufficient data ({len(forecasts_with_actuals)} samples)",
                metrics={'sample_count': len(forecasts_with_actuals)},
            )

        # Calculate MAPE
        errors = []
        for f in forecasts_with_actuals:
            if f.actual_eac_cents > 0:
                error = abs(f.predicted_eac_cents - f.actual_eac_cents) / f.actual_eac_cents
                errors.append(error)

        if not errors:
            return HealthCheckResult(
                check_name="prediction_accuracy",
                status="degraded",
                message="No valid samples for MAPE calculation",
                metrics={},
            )

        mape = sum(errors) / len(errors)

        if mape <= self.MAPE_WARNING_THRESHOLD:
            status = "healthy"
        elif mape <= self.MAPE_ERROR_THRESHOLD:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthCheckResult(
            check_name="prediction_accuracy",
            status=status,
            message=f"MAPE: {mape:.1%}",
            metrics={'mape': float(mape), 'sample_count': len(errors)},
        )

    def detect_model_drift(
        self,
        model_id: Optional[int] = None,
        lookback_days: int = 30,
    ) -> DriftReport:
        """
        Detect if model predictions are drifting from actuals.

        Args:
            model_id: Model to check (default: active model)
            lookback_days: Days to analyze

        Returns:
            DriftReport with drift analysis
        """
        if model_id is None:
            model = self.db.query(MLModelRegistry).filter(
                MLModelRegistry.is_production == True
            ).first()
            model_id = model.id if model else None

        if not model_id:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                affected_features=[],
                recommendation="No model to analyze",
            )

        # Get recent predictions with actuals
        cutoff = date.today() - timedelta(days=lookback_days)
        forecasts = self.db.query(ProjectForecast).filter(
            ProjectForecast.model_id == model_id,
            ProjectForecast.forecast_date >= cutoff,
            ProjectForecast.actual_eac_cents != None,
        ).all()

        if len(forecasts) < self.MIN_PREDICTIONS_FOR_ANALYSIS:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                affected_features=[],
                recommendation="Insufficient data for drift analysis",
                details={'sample_count': len(forecasts)},
            )

        # Calculate drift metrics
        recent_errors = []
        for f in forecasts:
            if f.actual_eac_cents > 0:
                error = (f.predicted_eac_cents - f.actual_eac_cents) / f.actual_eac_cents
                recent_errors.append(error)

        if not recent_errors:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                affected_features=[],
                recommendation="No valid samples",
            )

        # Check for systematic bias (mean error != 0)
        import numpy as np
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)

        # Drift score based on bias and variance
        drift_score = abs(mean_error) + std_error * 0.5

        drift_detected = drift_score > self.DRIFT_WARNING_THRESHOLD

        if drift_score > self.DRIFT_ERROR_THRESHOLD:
            recommendation = "Model retraining strongly recommended"
            self._send_alert(Alert(
                alert_type=AlertType.MODEL_DRIFT,
                severity=AlertSeverity.ERROR,
                message=f"Significant model drift detected (score: {drift_score:.2f})",
                details={'drift_score': drift_score, 'mean_error': mean_error},
                model_id=model_id,
            ))
        elif drift_detected:
            recommendation = "Consider model retraining"
            self._send_alert(Alert(
                alert_type=AlertType.MODEL_DRIFT,
                severity=AlertSeverity.WARNING,
                message=f"Model drift detected (score: {drift_score:.2f})",
                details={'drift_score': drift_score, 'mean_error': mean_error},
                model_id=model_id,
            ))
        else:
            recommendation = "Model performing within acceptable bounds"

        return DriftReport(
            drift_detected=drift_detected,
            drift_score=drift_score,
            affected_features=['predicted_eac'],
            recommendation=recommendation,
            details={
                'mean_error': float(mean_error),
                'std_error': float(std_error),
                'sample_count': len(recent_errors),
            },
        )

    def detect_data_drift(
        self,
        project_id: Optional[int] = None,
        lookback_days: int = 90,
    ) -> DriftReport:
        """
        Detect if input data distribution is drifting.

        Args:
            project_id: Project to check (None for all)
            lookback_days: Days to analyze

        Returns:
            DriftReport with data drift analysis
        """
        import numpy as np

        # Get recent vs historical features
        cutoff = date.today() - timedelta(days=lookback_days)
        midpoint = date.today() - timedelta(days=lookback_days // 2)

        query = self.db.query(CanonicalCostFeature)
        if project_id:
            query = query.filter(CanonicalCostFeature.project_id == project_id)

        historical = query.filter(
            CanonicalCostFeature.period_date < midpoint,
            CanonicalCostFeature.period_date >= cutoff,
        ).all()

        recent = query.filter(
            CanonicalCostFeature.period_date >= midpoint,
        ).all()

        if len(historical) < 10 or len(recent) < 10:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                affected_features=[],
                recommendation="Insufficient data for drift analysis",
            )

        # Compare distributions of key features
        affected_features = []
        max_drift = 0.0

        for feature_name in ['cost_per_sf_cents', 'budget_per_sf_cents']:
            hist_values = [getattr(f, feature_name) or 0 for f in historical]
            recent_values = [getattr(f, feature_name) or 0 for f in recent]

            # Simple drift detection: compare means and stds
            hist_mean = np.mean(hist_values)
            recent_mean = np.mean(recent_values)

            if hist_mean > 0:
                drift = abs(recent_mean - hist_mean) / hist_mean
                if drift > max_drift:
                    max_drift = drift
                if drift > self.DRIFT_WARNING_THRESHOLD:
                    affected_features.append(feature_name)

        drift_detected = max_drift > self.DRIFT_WARNING_THRESHOLD

        if drift_detected:
            recommendation = "Data distribution changing - monitor model performance"
            self._send_alert(Alert(
                alert_type=AlertType.DATA_DRIFT,
                severity=AlertSeverity.WARNING,
                message=f"Data drift detected in {len(affected_features)} features",
                details={'drift_score': max_drift, 'affected': affected_features},
                project_id=project_id,
            ))
        else:
            recommendation = "Data distribution stable"

        return DriftReport(
            drift_detected=drift_detected,
            drift_score=max_drift,
            affected_features=affected_features,
            recommendation=recommendation,
        )

    def check_retraining_needed(self) -> bool:
        """
        Check if model retraining is needed.

        Returns:
            True if retraining is recommended
        """
        # Check model drift
        drift_report = self.detect_model_drift()
        if drift_report.drift_score > self.DRIFT_ERROR_THRESHOLD:
            self._send_alert(Alert(
                alert_type=AlertType.RETRAINING_NEEDED,
                severity=AlertSeverity.WARNING,
                message="Model retraining recommended due to drift",
                details={'drift_score': drift_report.drift_score},
            ))
            return True

        # Check model age
        active_model = self.db.query(MLModelRegistry).filter(
            MLModelRegistry.is_production == True
        ).first()

        if active_model and active_model.created_at:
            age_days = (datetime.now() - active_model.created_at).days
            if age_days > 30:  # Model older than 30 days
                self._send_alert(Alert(
                    alert_type=AlertType.RETRAINING_NEEDED,
                    severity=AlertSeverity.INFO,
                    message=f"Model is {age_days} days old - consider retraining",
                    details={'model_age_days': age_days},
                ))
                return True

        return False

    def _send_alert(self, alert: Alert):
        """Send alert to all registered handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def _default_alert_handler(self, alert: Alert):
        """Default alert handler - logs the alert."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(
            log_level,
            f"[{alert.alert_type.value}] {alert.message}",
            extra={'alert_details': alert.details},
        )

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status summary.

        Returns:
            Status summary dictionary
        """
        health_checks = self.run_health_checks()

        # Determine overall status
        statuses = [hc.status for hc in health_checks]
        if 'unhealthy' in statuses:
            overall_status = 'unhealthy'
        elif 'degraded' in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'

        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'health_checks': [
                {
                    'name': hc.check_name,
                    'status': hc.status,
                    'message': hc.message,
                    'metrics': hc.metrics,
                }
                for hc in health_checks
            ],
        }

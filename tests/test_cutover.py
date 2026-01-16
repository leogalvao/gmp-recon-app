"""
Tests for Phase 5 Cutover System.

Tests:
1. Monitoring service health checks
2. Drift detection
3. Rollback operations
4. End-to-end cutover workflow
"""
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch


class TestMonitoringService:
    """Tests for MonitoringService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_health_checks_all_healthy(self, mock_db):
        """Test health checks return healthy status."""
        from app.domain.services import MonitoringService

        # Mock active model
        mock_model = MagicMock()
        mock_model.id = 1
        mock_model.name = "test_model"
        mock_model.version = "v1"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_model

        # Mock counts
        mock_db.query.return_value.filter.return_value.count.return_value = 10
        mock_db.query.return_value.scalar.return_value = date.today()

        service = MonitoringService(mock_db)

        # Get model availability check
        result = service._check_model_availability()

        assert result.status == "healthy"
        assert "Active model" in result.message

    def test_health_checks_no_model(self, mock_db):
        """Test health checks detect missing model."""
        from app.domain.services import MonitoringService

        mock_db.query.return_value.filter.return_value.first.return_value = None

        service = MonitoringService(mock_db)
        result = service._check_model_availability()

        assert result.status == "unhealthy"
        assert "No active model" in result.message

    def test_system_status_aggregation(self, mock_db):
        """Test system status aggregates health checks."""
        from app.domain.services import MonitoringService

        service = MonitoringService(mock_db)

        # Mock all checks to return healthy
        with patch.object(service, 'run_health_checks') as mock_checks:
            from app.domain.services import HealthCheckResult
            mock_checks.return_value = [
                HealthCheckResult("test1", "healthy", "OK"),
                HealthCheckResult("test2", "healthy", "OK"),
            ]

            status = service.get_system_status()

            assert status['overall_status'] == 'healthy'
            assert len(status['health_checks']) == 2

    def test_system_status_degraded(self, mock_db):
        """Test system status detects degraded state."""
        from app.domain.services import MonitoringService

        service = MonitoringService(mock_db)

        with patch.object(service, 'run_health_checks') as mock_checks:
            from app.domain.services import HealthCheckResult
            mock_checks.return_value = [
                HealthCheckResult("test1", "healthy", "OK"),
                HealthCheckResult("test2", "degraded", "Warning"),
            ]

            status = service.get_system_status()

            assert status['overall_status'] == 'degraded'


class TestRollbackService:
    """Tests for RollbackService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_rollback_project_success(self, mock_db):
        """Test successful project rollback."""
        from app.domain.services import RollbackService, RollbackReason

        mock_project = MagicMock()
        mock_project.id = 1
        mock_project.code = "TEST001"
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = []

        service = RollbackService(mock_db)

        with patch('app.domain.services.rollback_service.FeatureFlags') as mock_flags:
            result = service.rollback_project(
                project_id=1,
                reason=RollbackReason.MANUAL,
                triggered_by="test",
            )

            assert result.success
            assert result.flag_disabled
            mock_flags.disable_for_project.assert_called_once_with(1)

    def test_rollback_project_not_found(self, mock_db):
        """Test rollback fails for non-existent project."""
        from app.domain.services import RollbackService

        mock_db.query.return_value.get.return_value = None

        service = RollbackService(mock_db)
        result = service.rollback_project(project_id=999)

        assert not result.success
        assert "Project not found" in result.errors

    def test_batch_rollback(self, mock_db):
        """Test batch rollback of multiple projects."""
        from app.domain.services import RollbackService

        # Mock projects
        mock_project = MagicMock()
        mock_project.code = "TEST"
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = []

        service = RollbackService(mock_db)

        with patch('app.domain.services.rollback_service.FeatureFlags'):
            results = service.batch_rollback([1, 2, 3])

            assert len(results) == 3
            assert all(r.success for r in results.values())

    def test_rollback_history(self, mock_db):
        """Test rollback event history tracking."""
        from app.domain.services import RollbackService, RollbackReason

        mock_project = MagicMock()
        mock_project.code = "TEST"
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = []

        service = RollbackService(mock_db)

        with patch('app.domain.services.rollback_service.FeatureFlags'):
            service.rollback_project(1, RollbackReason.MANUAL, "test")
            service.rollback_project(2, RollbackReason.DATA_QUALITY, "system")

        history = service.get_rollback_history()

        assert len(history) == 2
        # Most recent first
        assert history[0].project_id == 2

    def test_can_reenable_check(self, mock_db):
        """Test re-enable eligibility check."""
        from app.domain.services import RollbackService

        mock_project = MagicMock()
        mock_project.data_quality_score = 0.8
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.count.return_value = 10

        service = RollbackService(mock_db)
        service._rollback_events = []  # Clear history

        can_enable, reasons = service.can_reenable(1)

        assert can_enable
        assert len(reasons) == 0

    def test_cannot_reenable_low_quality(self, mock_db):
        """Test re-enable blocked for low quality."""
        from app.domain.services import RollbackService

        mock_project = MagicMock()
        mock_project.data_quality_score = 0.5  # Below threshold
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.count.return_value = 10

        service = RollbackService(mock_db)

        can_enable, reasons = service.can_reenable(1)

        assert not can_enable
        assert any("quality" in r.lower() for r in reasons)


class TestDriftDetection:
    """Tests for drift detection."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_model_drift_insufficient_data(self, mock_db):
        """Test drift detection with insufficient data."""
        from app.domain.services import MonitoringService

        mock_model = MagicMock()
        mock_model.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_model
        mock_db.query.return_value.filter.return_value.all.return_value = []

        service = MonitoringService(mock_db)
        report = service.detect_model_drift()

        assert not report.drift_detected
        assert "Insufficient data" in report.recommendation

    def test_data_drift_detection(self, mock_db):
        """Test data drift detection."""
        from app.domain.services import MonitoringService

        # Create mock features with no drift
        mock_historical = [MagicMock(cost_per_sf_cents=1000, budget_per_sf_cents=1200) for _ in range(20)]
        mock_recent = [MagicMock(cost_per_sf_cents=1050, budget_per_sf_cents=1250) for _ in range(20)]

        mock_db.query.return_value.filter.return_value.all.side_effect = [mock_historical, mock_recent]

        service = MonitoringService(mock_db)
        report = service.detect_data_drift()

        # Small change should not trigger drift
        assert report.drift_score < 0.2


class TestEndToEndCutover:
    """End-to-end tests for cutover workflow."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_full_cutover_workflow(self, mock_db):
        """Test complete cutover workflow."""
        from app.domain.services import RollbackService
        from app.infrastructure.feature_flags import FeatureFlags

        # Mock project
        mock_project = MagicMock()
        mock_project.id = 1
        mock_project.code = "TEST001"
        mock_project.data_quality_score = 0.9
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.count.return_value = 10
        mock_db.query.return_value.filter.return_value.all.return_value = []

        # Step 1: Enable feature flags for project
        FeatureFlags.enable_for_project(1)

        # Step 2: Verify feature flags enabled
        assert FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(1)

        # Step 3: Rollback
        rollback_service = RollbackService(mock_db)
        rollback_result = rollback_service.rollback_project(1)

        assert rollback_result.success
        assert rollback_result.flag_disabled

        # Step 4: Verify feature flags disabled after rollback
        assert not FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(1)

    def test_cutover_blocked_low_quality(self, mock_db):
        """Test cutover blocked for low quality project."""
        from app.domain.services import ProjectCutoverService

        mock_project = MagicMock()
        mock_project.data_quality_score = 0.5  # Below threshold
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.count.return_value = 10

        service = ProjectCutoverService(mock_db)
        is_ready, issues = service.validate_cutover_readiness(1)

        assert not is_ready
        assert any("quality" in issue.lower() for issue in issues)

    def test_automatic_rollback_trigger(self, mock_db):
        """Test automatic rollback is triggered correctly."""
        from app.domain.services import RollbackService, RollbackReason

        mock_project = MagicMock()
        mock_project.data_quality_score = 0.4  # Very low
        mock_db.query.return_value.get.return_value = mock_project

        service = RollbackService(mock_db)
        reason = service.check_automatic_rollback_triggers(1)

        assert reason == RollbackReason.DATA_QUALITY


class TestAlertSystem:
    """Tests for alert system."""

    def test_alert_creation(self):
        """Test alert object creation."""
        from app.domain.services import Alert, AlertType, AlertSeverity

        alert = Alert(
            alert_type=AlertType.MODEL_DRIFT,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            details={'score': 0.15},
        )

        assert alert.alert_type == AlertType.MODEL_DRIFT
        assert alert.severity == AlertSeverity.WARNING
        assert alert.timestamp is not None

    def test_alert_handler_called(self):
        """Test custom alert handlers are called."""
        from app.domain.services import MonitoringService, Alert, AlertType, AlertSeverity

        mock_db = MagicMock()
        alerts_received = []

        def test_handler(alert):
            alerts_received.append(alert)

        service = MonitoringService(mock_db, alert_handlers=[test_handler])

        # Trigger an alert
        service._send_alert(Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.ERROR,
            message="Test",
            details={},
        ))

        assert len(alerts_received) == 1
        assert alerts_received[0].alert_type == AlertType.SYSTEM_ERROR


class TestCLICommands:
    """Tests for CLI commands structure."""

    def test_cutover_command_group_exists(self):
        """Test cutover CLI command group is defined."""
        from app.cli.cutover_commands import cutover

        assert cutover is not None
        assert cutover.name == 'cutover'

    def test_subcommands_registered(self):
        """Test all subcommands are registered."""
        from app.cli.cutover_commands import cutover

        command_names = [cmd for cmd in cutover.commands.keys()]

        assert 'execute' in command_names
        assert 'rollback' in command_names
        assert 'status' in command_names
        assert 'batch' in command_names
        assert 'flags' in command_names

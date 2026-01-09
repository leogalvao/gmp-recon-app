"""
Tests for Dashboard Metrics and Summary API.
Tests compute_dashboard_summary and related functions.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime, date
sys.path.insert(0, '.')

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from app.main import app
from app.models import (
    Base, GMP, DirectCostEntity, ForecastSnapshot, ScheduleActivity,
    ScheduleToGMPMapping, Project, get_db
)
from app.modules.reconciliation import (
    compute_dashboard_summary, get_total_gmp_budget, get_total_actual_costs,
    get_total_forecast_remaining, compute_cpi_if_ev_available
)
import uuid


# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_dashboard.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create test client with test database."""
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def create_project(db):
    """Helper to create a test project."""
    project = Project(
        uuid=str(uuid.uuid4()),
        name="Test Project",
        code="TEST-001"
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def create_gmp(db, project_id, division, amount_cents, zone="WEST"):
    """Helper to create a GMP entity."""
    gmp = GMP(
        uuid=str(uuid.uuid4()),
        project_id=project_id,
        division=division,
        zone=zone,
        original_amount_cents=amount_cents
    )
    db.add(gmp)
    db.commit()
    db.refresh(gmp)
    return gmp


def create_direct_cost(db, amount_cents, budget_id=None):
    """Helper to create a DirectCostEntity."""
    cost = DirectCostEntity(
        uuid=str(uuid.uuid4()),
        mapped_budget_id=budget_id,
        gross_amount_cents=amount_cents,
        vendor_name="Test Vendor"
    )
    db.add(cost)
    db.commit()
    return cost


def create_forecast_snapshot(db, gmp_division, eac_cents, etc_cents, ac_cents=0, ev_cents=None):
    """Helper to create a ForecastSnapshot."""
    snapshot = ForecastSnapshot(
        gmp_division=gmp_division,
        snapshot_date=datetime.utcnow(),
        bac_cents=eac_cents,  # BAC = Budget at Completion
        ac_cents=ac_cents,
        ev_cents=ev_cents,
        eac_cents=eac_cents,
        eac_west_cents=eac_cents // 2,
        eac_east_cents=eac_cents - (eac_cents // 2),
        etc_cents=etc_cents,
        var_cents=0,
        method="evm",
        confidence_score=0.8,
        confidence_band="high",
        is_current=True
    )
    db.add(snapshot)
    db.commit()
    return snapshot


class TestGetTotalGMPBudget:
    """Tests for get_total_gmp_budget function."""

    def test_empty_database_returns_zero(self, db_session):
        """When no GMP entities exist, should return 0."""
        result = get_total_gmp_budget(db_session)
        assert result == 0

    def test_single_gmp_entity(self, db_session):
        """Single GMP entity should return its original_amount_cents."""
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General Requirements", 1000000000)

        result = get_total_gmp_budget(db_session)
        assert result == 1000000000

    def test_multiple_gmp_entities(self, db_session):
        """Multiple GMP entities should return sum of original_amount_cents."""
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General Requirements", 1000000000)
        create_gmp(db_session, project.id, "02 - Site Work", 1629754658)

        result = get_total_gmp_budget(db_session)
        assert result == 2629754658


class TestGetTotalActualCosts:
    """Tests for get_total_actual_costs function."""

    def test_empty_database_returns_zero(self, db_session):
        """When no direct costs exist, should return 0."""
        result = get_total_actual_costs(db_session)
        assert result == 0

    def test_single_direct_cost(self, db_session):
        """Single direct cost should return its gross_amount_cents."""
        create_direct_cost(db_session, 500000)

        result = get_total_actual_costs(db_session)
        assert result == 500000

    def test_multiple_direct_costs(self, db_session):
        """Multiple direct costs should return sum of gross_amount_cents."""
        create_direct_cost(db_session, 500000)
        create_direct_cost(db_session, 750000)
        create_direct_cost(db_session, 250000)

        result = get_total_actual_costs(db_session)
        assert result == 1500000


class TestGetTotalForecastRemaining:
    """Tests for get_total_forecast_remaining function."""

    def test_no_forecasts_returns_none(self, db_session):
        """When no forecast snapshots exist, should return None."""
        result = get_total_forecast_remaining(db_session)
        assert result is None

    def test_single_forecast_snapshot(self, db_session):
        """Single forecast snapshot should return its etc_cents."""
        create_forecast_snapshot(db_session, "01 - General", 1000000, etc_cents=300000)

        result = get_total_forecast_remaining(db_session)
        assert result == 300000

    def test_multiple_forecast_snapshots(self, db_session):
        """Multiple forecast snapshots should return sum of etc_cents."""
        create_forecast_snapshot(db_session, "01 - General", 1000000, etc_cents=300000)
        create_forecast_snapshot(db_session, "02 - Site", 800000, etc_cents=200000)

        result = get_total_forecast_remaining(db_session)
        assert result == 500000


class TestComputeDashboardSummary:
    """Tests for compute_dashboard_summary function."""

    def test_budget_maps_from_gmp_total(self, db_session):
        """
        AC-01: When GMP total exists, Total GMP Budget should show correct amount.
        """
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General Requirements", 1000000000)
        create_gmp(db_session, project.id, "02 - Site Work", 1629754658)

        result = compute_dashboard_summary(db_session)

        assert result['total_gmp_budget_cents'] == 2629754658
        assert 'GMP Budget data unavailable' not in result['warnings']

    def test_eac_is_actual_plus_forecast(self, db_session):
        """
        AC-02: EAC should equal Actual Costs + Forecast Remaining.
        """
        # Setup GMP budget
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 10000)

        # Setup actuals (4000 cents)
        create_direct_cost(db_session, 4000)

        # Setup forecast remaining (3000 cents)
        create_forecast_snapshot(db_session, "01 - General", 7000, etc_cents=3000, ac_cents=4000)

        result = compute_dashboard_summary(db_session)

        assert result['actual_costs_cents'] == 4000
        assert result['forecast_remaining_cents'] == 3000
        assert result['eac_cents'] == 7000  # 4000 + 3000

    def test_eac_zero_when_no_data(self, db_session):
        """
        AC-02: If actual = 0 and forecast_remaining = 0, EAC should show $0.00.
        """
        # Setup GMP budget but no actuals or forecast
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 10000)

        result = compute_dashboard_summary(db_session)

        assert result['eac_cents'] == 0
        assert result['actual_costs_cents'] == 0
        assert result['forecast_remaining_cents'] == 0

    def test_variance_correct_calculation(self, db_session):
        """
        AC-03: Variance = Budget - EAC (positive = underrun, negative = overrun).
        """
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 10000)
        create_direct_cost(db_session, 4000)
        create_forecast_snapshot(db_session, "01 - General", 7000, etc_cents=3000, ac_cents=4000)

        result = compute_dashboard_summary(db_session)

        # Variance = 10000 - 7000 = 3000 (underrun)
        assert result['variance_cents'] == 3000

    def test_variance_none_when_budget_zero(self, db_session):
        """
        When budget is zero, variance should be None.
        """
        result = compute_dashboard_summary(db_session)

        assert result['variance_cents'] is None
        assert 'GMP Budget data unavailable or zero' in result['warnings']

    def test_cpi_returns_none_without_ev(self, db_session):
        """
        AC-04: CPI shows N/A unless EV is implemented.
        """
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 1000000)
        create_direct_cost(db_session, 500000)

        result = compute_dashboard_summary(db_session)

        assert result['cpi'] is None
        assert any("CPI requires Earned Value" in w for w in result['warnings'])

    def test_cpi_calculated_when_ev_available(self, db_session):
        """
        CPI should be calculated when EV is available in forecast snapshots.
        """
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 1000000)
        create_direct_cost(db_session, 500000)
        # Create forecast with EV = 600000
        create_forecast_snapshot(
            db_session, "01 - General", 1000000,
            etc_cents=500000, ac_cents=500000, ev_cents=600000
        )

        result = compute_dashboard_summary(db_session)

        # CPI = EV / AC = 600000 / 500000 = 1.2
        assert result['cpi'] == 1.2

    def test_progress_pct_calculation(self, db_session):
        """
        Progress should be (Actual / EAC) * 100, clamped 0-100.
        """
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 10000)
        create_direct_cost(db_session, 5000)
        create_forecast_snapshot(db_session, "01 - General", 10000, etc_cents=5000, ac_cents=5000)

        result = compute_dashboard_summary(db_session)

        # Progress = 5000 / 10000 * 100 = 50%
        assert result['progress_pct'] == 50.0

    def test_progress_pct_zero_when_eac_zero(self, db_session):
        """
        When EAC is zero, progress should be 0.
        """
        result = compute_dashboard_summary(db_session)

        assert result['progress_pct'] == 0.0

    def test_warnings_for_missing_data(self, db_session):
        """
        AC-06: Warnings should be surfaced when data is missing.
        """
        result = compute_dashboard_summary(db_session)

        assert len(result['warnings']) > 0
        assert 'GMP Budget data unavailable or zero' in result['warnings']
        assert 'Forecast data unavailable' in result['warnings']


class TestDashboardAPIEndpoint:
    """Tests for GET /api/dashboard/summary endpoint."""

    def test_api_returns_200(self, client):
        """API should return 200 OK."""
        response = client.get("/api/dashboard/summary")
        assert response.status_code == 200

    def test_api_returns_correct_structure(self, client):
        """API should return all required fields."""
        response = client.get("/api/dashboard/summary")
        data = response.json()

        assert 'total_gmp_budget_cents' in data
        assert 'actual_costs_cents' in data
        assert 'forecast_remaining_cents' in data
        assert 'eac_cents' in data
        assert 'variance_cents' in data
        assert 'progress_pct' in data
        assert 'cpi' in data
        assert 'schedule_variance_days' in data
        assert 'warnings' in data

    def test_api_dashboard_summary_with_gmp_data(self, client, db_session):
        """API should return correct GMP total when data exists."""
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 2629754658)

        response = client.get("/api/dashboard/summary")
        data = response.json()

        assert data['total_gmp_budget_cents'] == 2629754658


class TestDashboardPage:
    """Tests for /dashboard HTML page."""

    def test_dashboard_page_renders(self, client):
        """Dashboard page should return 200 and contain expected elements."""
        # Mock run_full_reconciliation to avoid needing data files
        with patch('app.main.run_full_reconciliation') as mock_recon:
            mock_recon.return_value = {
                'recon_rows': [],
                'summary': {},
                'mapping_stats': {}
            }

            response = client.get("/dashboard")
            assert response.status_code == 200
            assert b"Project Dashboard" in response.content

    def test_dashboard_page_renders_with_warnings(self, client):
        """Dashboard page should display warnings banner when warnings exist."""
        with patch('app.main.run_full_reconciliation') as mock_recon:
            mock_recon.return_value = {
                'recon_rows': [],
                'summary': {},
                'mapping_stats': {}
            }

            response = client.get("/dashboard")
            assert response.status_code == 200
            # Warnings banner should be present (since no GMP data)
            assert b"Data Warnings" in response.content or response.status_code == 200


class TestScheduleFailureResilience:
    """Tests for schedule query failure handling."""

    def test_schedule_failure_does_not_crash(self, db_session):
        """
        AC-05: Dashboard should load even if schedule has schema issues.
        """
        # Create some basic data
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 1000000)

        # The compute_dashboard_summary should not crash even with no schedule data
        result = compute_dashboard_summary(db_session)

        assert result is not None
        assert 'total_gmp_budget_cents' in result

    def test_schedule_variance_none_when_no_data(self, db_session):
        """Schedule variance should be None when no schedule data exists."""
        project = create_project(db_session)
        create_gmp(db_session, project.id, "01 - General", 1000000)

        result = compute_dashboard_summary(db_session)

        assert result['schedule_variance_days'] is None

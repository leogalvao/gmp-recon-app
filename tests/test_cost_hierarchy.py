"""
Unit Tests for Cost Management Hierarchy.

Tests business rules:
- GMP immutability
- Budget ceiling constraints
- Vertical reconciliation invariants
"""
import pytest
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, GMP, BudgetEntity, DirectCostEntity, Project
from app.domain.exceptions import (
    ImmutableFieldError,
    GMPCeilingExceededError,
    BudgetUnderflowError,
    BudgetHasMappedCostsError,
    GMPNotFoundError,
    BudgetNotFoundError,
)
from app.infrastructure.repositories import (
    GMPRepository,
    BudgetRepository,
    DirectCostRepository,
)
from app.domain.services import (
    CostAggregationService,
    BudgetValidationService,
)
from app.domain.events.handlers import (
    validate_gmp_ceiling,
    validate_budget_not_below_actual,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def test_db():
    """Create a test database with fresh tables for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a test project
    project = Project(
        uuid=str(uuid.uuid4()),
        name="Test Project",
        code="TEST-001"
    )
    session.add(project)
    session.commit()

    yield session, project

    session.close()


@pytest.fixture
def sample_gmp(test_db):
    """Create a sample GMP for testing."""
    session, project = test_db
    gmp = GMP(
        uuid=str(uuid.uuid4()),
        project_id=project.id,
        division="Division 03 - Concrete",
        zone="EAST",
        original_amount_cents=10000000,  # $100,000
        description="Test GMP"
    )
    session.add(gmp)
    session.commit()
    return gmp


@pytest.fixture
def sample_budget(test_db, sample_gmp):
    """Create a sample budget for testing."""
    session, _ = test_db
    budget = BudgetEntity(
        uuid=str(uuid.uuid4()),
        gmp_id=sample_gmp.id,
        cost_code="03-2100",
        description="Concrete Foundations",
        zone="EAST",
        current_budget_cents=5000000  # $50,000
    )
    session.add(budget)
    session.commit()
    return budget


@pytest.fixture
def sample_direct_cost(test_db, sample_budget):
    """Create a sample direct cost for testing."""
    session, _ = test_db
    cost = DirectCostEntity(
        uuid=str(uuid.uuid4()),
        mapped_budget_id=sample_budget.id,
        vendor_name="ABC Concrete",
        description="Foundation Pour - Phase 1",
        transaction_date=date(2024, 6, 15),
        gross_amount_cents=1500000,  # $15,000
        zone="EAST"
    )
    session.add(cost)
    session.commit()
    return cost


# =============================================================================
# GMP Immutability Tests
# =============================================================================

class TestGMPImmutability:
    """Test GMP amount immutability."""

    def test_gmp_amount_positive_required(self, test_db):
        """Test that GMP amount must be positive."""
        session, project = test_db
        repo = GMPRepository(session)

        with pytest.raises(ValueError, match="positive"):
            repo.create(
                project_id=project.id,
                division="Division 04",
                zone="EAST",
                original_amount_cents=0
            )

    def test_gmp_amount_positive_negative_rejected(self, test_db):
        """Test that negative GMP amount is rejected."""
        session, project = test_db
        repo = GMPRepository(session)

        with pytest.raises(ValueError, match="positive"):
            repo.create(
                project_id=project.id,
                division="Division 04",
                zone="EAST",
                original_amount_cents=-1000
            )

    def test_gmp_description_update_allowed(self, test_db, sample_gmp):
        """Test that GMP description can be updated."""
        session, _ = test_db
        repo = GMPRepository(session)

        original_desc = sample_gmp.description
        updated = repo.update_description(sample_gmp.id, "Updated description")
        session.commit()

        assert updated.description == "Updated description"
        assert updated.description != original_desc

    def test_gmp_unique_constraint(self, test_db, sample_gmp):
        """Test that duplicate GMP division/zone raises error."""
        session, project = test_db
        repo = GMPRepository(session)

        from app.domain.exceptions import DuplicateGMPError
        with pytest.raises(DuplicateGMPError):
            repo.create(
                project_id=project.id,
                division=sample_gmp.division,
                zone=sample_gmp.zone,
                original_amount_cents=5000000
            )


# =============================================================================
# Budget Ceiling Constraint Tests
# =============================================================================

class TestBudgetCeilingConstraint:
    """Test GMP ceiling constraint enforcement."""

    def test_budget_within_ceiling_allowed(self, test_db, sample_gmp):
        """Test that budget within ceiling is allowed."""
        session, _ = test_db
        repo = BudgetRepository(session)

        budget = repo.create(
            gmp_id=sample_gmp.id,
            cost_code="03-2200",
            current_budget_cents=3000000,  # $30,000 (within $100,000 ceiling)
            zone="EAST"
        )
        session.commit()

        assert budget.current_budget_cents == 3000000

    def test_budget_exceeding_ceiling_rejected(self, test_db, sample_gmp):
        """Test that budget exceeding ceiling is rejected."""
        session, _ = test_db
        repo = BudgetRepository(session)

        with pytest.raises(GMPCeilingExceededError):
            repo.create(
                gmp_id=sample_gmp.id,
                cost_code="03-2200",
                current_budget_cents=15000000,  # $150,000 (exceeds $100,000)
                zone="EAST"
            )

    def test_budget_update_exceeding_ceiling_rejected(self, test_db, sample_gmp, sample_budget):
        """Test that budget update exceeding ceiling is rejected."""
        session, _ = test_db
        repo = BudgetRepository(session)

        with pytest.raises(GMPCeilingExceededError):
            repo.update_amount(sample_budget.id, 12000000)  # $120,000 exceeds ceiling

    def test_budget_sum_ceiling_calculation(self, test_db, sample_gmp):
        """Test that sum of budgets is correctly calculated against ceiling."""
        session, _ = test_db
        repo = BudgetRepository(session)
        validation = BudgetValidationService(session)

        # Create first budget: $50,000
        budget1 = repo.create(
            gmp_id=sample_gmp.id,
            cost_code="03-2100",
            current_budget_cents=5000000,
            zone="EAST"
        )
        session.commit()

        # Create second budget: $40,000 (total: $90,000, within $100,000)
        budget2 = repo.create(
            gmp_id=sample_gmp.id,
            cost_code="03-2200",
            current_budget_cents=4000000,
            zone="EAST"
        )
        session.commit()

        # Third budget of $15,000 would exceed ceiling
        with pytest.raises(GMPCeilingExceededError):
            repo.create(
                gmp_id=sample_gmp.id,
                cost_code="03-2300",
                current_budget_cents=1500000,  # Would make total $105,000
                zone="EAST"
            )


# =============================================================================
# Budget Underflow Tests
# =============================================================================

class TestBudgetUnderflow:
    """Test that budget cannot be reduced below actual spent."""

    def test_budget_not_below_actual(self, test_db, sample_budget, sample_direct_cost):
        """Test that budget cannot be reduced below actual cost."""
        session, _ = test_db
        repo = BudgetRepository(session)

        # Direct cost is $15,000, try to reduce budget to $10,000
        with pytest.raises(BudgetUnderflowError):
            repo.update_amount(sample_budget.id, 1000000)  # $10,000

    def test_budget_at_actual_allowed(self, test_db, sample_budget, sample_direct_cost):
        """Test that budget can be set equal to actual."""
        session, _ = test_db
        repo = BudgetRepository(session)

        # Set budget to exactly $15,000 (same as actual)
        updated = repo.update_amount(sample_budget.id, 1500000)
        session.commit()

        assert updated.current_budget_cents == 1500000


# =============================================================================
# Budget Deletion Tests
# =============================================================================

class TestBudgetDeletion:
    """Test budget deletion constraints."""

    def test_budget_with_costs_cannot_be_deleted(self, test_db, sample_budget, sample_direct_cost):
        """Test that budget with mapped costs cannot be deleted."""
        session, _ = test_db
        repo = BudgetRepository(session)

        with pytest.raises(BudgetHasMappedCostsError):
            repo.delete_budget(sample_budget.id)

    def test_budget_without_costs_can_be_deleted(self, test_db, sample_gmp):
        """Test that budget without mapped costs can be deleted."""
        session, _ = test_db
        repo = BudgetRepository(session)

        # Create new budget without costs
        budget = repo.create(
            gmp_id=sample_gmp.id,
            cost_code="03-9999",
            current_budget_cents=1000000,
            zone="EAST"
        )
        session.commit()

        budget_id = budget.id
        repo.delete_budget(budget_id)
        session.commit()

        assert repo.get_by_id(budget_id) is None


# =============================================================================
# Vertical Reconciliation Tests
# =============================================================================

class TestVerticalReconciliation:
    """Test vertical reconciliation invariants."""

    def test_budget_actual_equals_sum_of_costs(self, test_db, sample_budget, sample_direct_cost):
        """Test that budget actual equals sum of direct costs."""
        session, _ = test_db
        aggregation = CostAggregationService(session)

        # Add another cost
        cost2 = DirectCostEntity(
            uuid=str(uuid.uuid4()),
            mapped_budget_id=sample_budget.id,
            vendor_name="XYZ Concrete",
            description="Foundation Pour - Phase 2",
            transaction_date=date(2024, 6, 20),
            gross_amount_cents=500000,  # $5,000
            zone="EAST"
        )
        session.add(cost2)
        session.commit()

        # Calculate actual
        actual = aggregation.recalculate_budget_actual(sample_budget.id)

        # Expected: $15,000 + $5,000 = $20,000
        assert actual == 2000000

    def test_gmp_total_equals_sum_of_budget_actuals(self, test_db, sample_gmp, sample_budget, sample_direct_cost):
        """Test that GMP total equals sum of budget actuals."""
        session, _ = test_db
        aggregation = CostAggregationService(session)

        totals = aggregation.recalculate_gmp_totals(sample_gmp.id)

        assert totals['total_actual_cents'] == sample_direct_cost.gross_amount_cents

    def test_reconciliation_validation_passes(self, test_db, sample_gmp, sample_budget, sample_direct_cost):
        """Test that reconciliation validation passes for valid data."""
        session, _ = test_db
        aggregation = CostAggregationService(session)

        is_valid, errors = aggregation.validate_vertical_reconciliation(sample_gmp.id)

        assert is_valid is True
        assert len(errors) == 0


# =============================================================================
# Period Aggregation Tests
# =============================================================================

class TestPeriodAggregation:
    """Test temporal aggregation functions."""

    def test_weekly_totals(self, test_db, sample_budget, sample_direct_cost):
        """Test weekly cost aggregation."""
        session, _ = test_db
        aggregation = CostAggregationService(session)

        # Add costs on different weeks
        cost2 = DirectCostEntity(
            uuid=str(uuid.uuid4()),
            mapped_budget_id=sample_budget.id,
            vendor_name="XYZ",
            description="Week 2 Cost",
            transaction_date=date(2024, 6, 22),  # Different week
            gross_amount_cents=200000,
            zone="EAST"
        )
        session.add(cost2)
        session.commit()

        totals = aggregation.get_weekly_totals(
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30)
        )

        assert len(totals) >= 1

    def test_monthly_totals(self, test_db, sample_budget, sample_direct_cost):
        """Test monthly cost aggregation."""
        session, _ = test_db
        aggregation = CostAggregationService(session)

        totals = aggregation.get_monthly_totals(
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30)
        )

        assert len(totals) >= 1
        assert any(t['period_id'] == '2024-06' for t in totals)

    def test_week_proration(self, test_db):
        """Test week proration across months."""
        session, _ = test_db
        aggregation = CostAggregationService(session)

        # Week spanning Jan 28 - Feb 3
        prorated = aggregation.prorate_spanning_week(
            week_start=date(2024, 1, 28),  # Sunday
            week_end=date(2024, 2, 3),      # Saturday
            week_amount_cents=700000        # $7,000
        )

        # Should have entries for both months
        assert '2024-01' in prorated
        assert '2024-02' in prorated

        # Amounts should sum to total
        total = sum(prorated.values())
        assert total == 700000


# =============================================================================
# Repository Tests
# =============================================================================

class TestRepositories:
    """Test repository pattern implementations."""

    def test_gmp_repository_get_by_division(self, test_db, sample_gmp):
        """Test getting GMP by division."""
        session, _ = test_db
        repo = GMPRepository(session)

        gmps = repo.get_by_division(sample_gmp.division)

        assert len(gmps) >= 1
        assert gmps[0].division == sample_gmp.division

    def test_budget_repository_get_by_gmp(self, test_db, sample_gmp, sample_budget):
        """Test getting budgets by GMP."""
        session, _ = test_db
        repo = BudgetRepository(session)

        budgets = repo.get_by_gmp(sample_gmp.id)

        assert len(budgets) >= 1
        assert budgets[0].gmp_id == sample_gmp.id

    def test_direct_cost_repository_get_unmapped(self, test_db):
        """Test getting unmapped direct costs."""
        session, _ = test_db
        repo = DirectCostRepository(session)

        # Create unmapped cost
        cost = DirectCostEntity(
            uuid=str(uuid.uuid4()),
            mapped_budget_id=None,
            vendor_name="Unmapped Vendor",
            description="Unmapped Cost",
            transaction_date=date.today(),
            gross_amount_cents=100000
        )
        session.add(cost)
        session.commit()

        unmapped = repo.get_unmapped()

        assert len(unmapped) >= 1
        assert any(c.vendor_name == "Unmapped Vendor" for c in unmapped)

    def test_direct_cost_bulk_map(self, test_db, sample_budget):
        """Test bulk mapping of direct costs."""
        session, _ = test_db
        repo = DirectCostRepository(session)

        # Create multiple unmapped costs
        costs = []
        for i in range(3):
            cost = DirectCostEntity(
                uuid=str(uuid.uuid4()),
                mapped_budget_id=None,
                vendor_name=f"Vendor {i}",
                description=f"Cost {i}",
                transaction_date=date.today(),
                gross_amount_cents=100000 * (i + 1)
            )
            session.add(cost)
            costs.append(cost)
        session.commit()

        # Bulk map
        mappings = [
            {'direct_cost_id': c.id, 'budget_id': sample_budget.id}
            for c in costs
        ]
        updated, affected = repo.bulk_map(mappings)
        session.commit()

        assert updated == 3
        assert sample_budget.id in affected


# =============================================================================
# Validation Service Tests
# =============================================================================

class TestValidationService:
    """Test budget validation service."""

    def test_validate_budget_create_valid(self, test_db, sample_gmp):
        """Test validation for valid budget creation."""
        session, _ = test_db
        validation = BudgetValidationService(session)

        is_valid, errors = validation.validate_budget_create(
            gmp_id=sample_gmp.id,
            amount_cents=3000000,
            cost_code="03-2100"
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_budget_create_exceeds_ceiling(self, test_db, sample_gmp):
        """Test validation for budget exceeding ceiling."""
        session, _ = test_db
        validation = BudgetValidationService(session)

        is_valid, errors = validation.validate_budget_create(
            gmp_id=sample_gmp.id,
            amount_cents=15000000,  # Exceeds $100,000
            cost_code="03-2100"
        )

        assert is_valid is False
        assert len(errors) > 0

    def test_validation_summary(self, test_db, sample_gmp, sample_budget, sample_direct_cost):
        """Test validation summary generation."""
        session, _ = test_db
        validation = BudgetValidationService(session)

        summary = validation.get_validation_summary(sample_gmp.id)

        assert 'ceiling_cents' in summary
        assert 'total_budgeted_cents' in summary
        assert 'budgets' in summary
        assert len(summary['budgets']) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

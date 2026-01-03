"""
Tests for the GMP Allocation Override API endpoints.
"""
import pytest
import sys
sys.path.insert(0, '.')

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app, get_db
from app.models import Base, GMPAllocationOverride, AllocationChangeLog


# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_allocation.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client():
    """Create test client with test database."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Override dependency
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    # Cleanup
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clean up allocation overrides before each test."""
    db = TestingSessionLocal()
    try:
        db.query(GMPAllocationOverride).delete()
        db.query(AllocationChangeLog).delete()
        db.commit()
    finally:
        db.close()
    yield


class TestGetAllocation:
    """Tests for GET /api/gmp/allocations/{gmp_division}"""

    def test_get_allocation_no_override(self, client):
        """Test getting allocation when no override exists."""
        response = client.get("/api/gmp/allocations/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert data['gmp_division'] == 'Concrete'
        assert data['has_override'] == False
        assert data['override_west'] is None
        assert data['override_east'] is None
        assert 'computed_west' in data
        assert 'computed_east' in data
        assert 'computed_total' in data
        assert 'gmp_total' in data

    def test_get_allocation_with_override(self, client):
        """Test getting allocation when override exists."""
        # First create an override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 2000000,
                "notes": "Test override"
            }
        )

        response = client.get("/api/gmp/allocations/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert data['gmp_division'] == 'Concrete'
        assert data['has_override'] == True
        assert data['override_west'] == 1000000
        assert data['override_east'] == 2000000
        assert data['notes'] == "Test override"

    def test_get_allocation_url_encoded_division(self, client):
        """Test getting allocation with URL-encoded division name."""
        response = client.get("/api/gmp/allocations/Aluminum%20%26%20Glass")
        assert response.status_code == 200

        data = response.json()
        assert data['gmp_division'] == 'Aluminum & Glass'

    def test_get_allocation_nonexistent_division(self, client):
        """Test getting allocation for division with no data."""
        response = client.get("/api/gmp/allocations/NonexistentDivision")
        assert response.status_code == 200

        data = response.json()
        assert data['computed_west'] == 0
        assert data['computed_east'] == 0


class TestSaveAllocation:
    """Tests for POST /api/gmp/allocations/{gmp_division}"""

    def test_save_allocation_new_override(self, client):
        """Test creating a new allocation override."""
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 5000000,
                "amount_east_cents": 5000000,
                "notes": "Initial override"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data['success'] == True
        assert data['gmp_division'] == 'Concrete'
        assert data['amount_west_cents'] == 5000000
        assert data['amount_east_cents'] == 5000000

    def test_save_allocation_update_override(self, client):
        """Test updating an existing allocation override."""
        # Create initial override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 2000000,
                "notes": "Initial"
            }
        )

        # Update override
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1500000,
                "amount_east_cents": 1500000,
                "notes": "Updated"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data['success'] == True
        assert data['amount_west_cents'] == 1500000
        assert data['amount_east_cents'] == 1500000

    def test_save_allocation_creates_audit_log(self, client):
        """Test that saving allocation creates audit log entries."""
        client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 2000000,
                "notes": "Test audit"
            }
        )

        # Check audit log
        response = client.get("/api/gmp/allocation-history/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert len(data['history']) >= 2  # At least west and east entries

        fields_changed = [h['field'] for h in data['history']]
        assert 'amount_west' in fields_changed
        assert 'amount_east' in fields_changed

    def test_save_allocation_with_special_characters(self, client):
        """Test saving allocation for division with special characters."""
        response = client.post(
            "/api/gmp/allocations/Doors%2C%20Frames%2C%20%26%20Hardware",
            json={
                "amount_west_cents": 500000,
                "amount_east_cents": 500000,
                "notes": "Special chars test"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data['success'] == True
        assert data['gmp_division'] == 'Doors, Frames, & Hardware'

    def test_save_allocation_zero_values(self, client):
        """Test saving allocation with zero values."""
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 0,
                "amount_east_cents": 0,
                "notes": "Zero override"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data['success'] == True
        assert data['amount_west_cents'] == 0
        assert data['amount_east_cents'] == 0


class TestClearAllocation:
    """Tests for DELETE /api/gmp/allocations/{gmp_division}"""

    def test_clear_allocation_existing_override(self, client):
        """Test clearing an existing allocation override."""
        # First create an override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 2000000,
                "notes": "To be cleared"
            }
        )

        # Clear it
        response = client.delete("/api/gmp/allocations/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert data['success'] == True
        assert data['message'] == 'Override cleared'

        # Verify it's cleared
        get_response = client.get("/api/gmp/allocations/Concrete")
        get_data = get_response.json()
        assert get_data['has_override'] == False

    def test_clear_allocation_no_override(self, client):
        """Test clearing when no override exists."""
        response = client.delete("/api/gmp/allocations/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert data['success'] == False
        assert 'No override' in data['message']

    def test_clear_allocation_creates_audit_log(self, client):
        """Test that clearing allocation creates audit log entry."""
        # Create override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 2000000,
                "notes": "Will be cleared"
            }
        )

        # Clear it
        client.delete("/api/gmp/allocations/Concrete")

        # Check audit log has clear entry
        response = client.get("/api/gmp/allocation-history/Concrete")
        data = response.json()

        # Should have entries for create and clear
        fields = [h['field'] for h in data['history']]
        assert 'override_cleared' in fields


class TestAllocationHistory:
    """Tests for GET /api/gmp/allocation-history/{gmp_division}"""

    def test_get_history_empty(self, client):
        """Test getting history when no changes exist."""
        response = client.get("/api/gmp/allocation-history/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert data['gmp_division'] == 'Concrete'
        assert data['history'] == []

    def test_get_history_with_changes(self, client):
        """Test getting history with multiple changes."""
        # Create override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={"amount_west_cents": 1000000, "amount_east_cents": 2000000, "notes": "First"}
        )

        # Update override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={"amount_west_cents": 1500000, "amount_east_cents": 2500000, "notes": "Second"}
        )

        response = client.get("/api/gmp/allocation-history/Concrete")
        assert response.status_code == 200

        data = response.json()
        assert len(data['history']) >= 4  # At least 2 creates + 2 updates

    def test_get_history_fields(self, client):
        """Test that history entries have required fields."""
        client.post(
            "/api/gmp/allocations/Concrete",
            json={"amount_west_cents": 1000000, "amount_east_cents": 2000000, "notes": "Test"}
        )

        response = client.get("/api/gmp/allocation-history/Concrete")
        data = response.json()

        if data['history']:
            entry = data['history'][0]
            assert 'field' in entry
            assert 'old_value' in entry
            assert 'new_value' in entry
            assert 'changed_by' in entry
            assert 'changed_at' in entry

    def test_get_history_chronological_order(self, client):
        """Test that history is returned in chronological order."""
        # Create multiple changes
        for i in range(3):
            client.post(
                "/api/gmp/allocations/Concrete",
                json={
                    "amount_west_cents": (i + 1) * 1000000,
                    "amount_east_cents": (i + 1) * 1000000,
                    "notes": f"Change {i + 1}"
                }
            )

        response = client.get("/api/gmp/allocation-history/Concrete")
        data = response.json()

        # History should have multiple entries
        assert len(data['history']) >= 3


class TestAllocationValidation:
    """Tests for allocation validation logic."""

    def test_computed_total_equals_sum(self, client):
        """Test that computed_total equals computed_west + computed_east."""
        response = client.get("/api/gmp/allocations/Concrete")
        data = response.json()

        assert data['computed_total'] == data['computed_west'] + data['computed_east']

    def test_override_preserves_computed_values(self, client):
        """Test that overrides don't affect computed values."""
        # Get original computed values
        original = client.get("/api/gmp/allocations/Concrete").json()

        # Create override
        client.post(
            "/api/gmp/allocations/Concrete",
            json={"amount_west_cents": 9999999, "amount_east_cents": 8888888, "notes": "Override"}
        )

        # Computed values should be unchanged
        after = client.get("/api/gmp/allocations/Concrete").json()
        assert after['computed_west'] == original['computed_west']
        assert after['computed_east'] == original['computed_east']
        assert after['computed_total'] == original['computed_total']


class TestAllocationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_large_values(self, client):
        """Test handling of large cent values."""
        large_value = 99999999999  # ~$1 billion
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": large_value,
                "amount_east_cents": large_value,
                "notes": "Large value test"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data['amount_west_cents'] == large_value

    def test_negative_values(self, client):
        """Test handling of negative values (credits)."""
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": -100000,
                "amount_east_cents": 100000,
                "notes": "Negative value test"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data['amount_west_cents'] == -100000

    def test_empty_notes(self, client):
        """Test saving with empty notes."""
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 1000000,
                "notes": ""
            }
        )
        assert response.status_code == 200

    def test_missing_notes_field(self, client):
        """Test saving without notes field."""
        response = client.post(
            "/api/gmp/allocations/Concrete",
            json={
                "amount_west_cents": 1000000,
                "amount_east_cents": 1000000
            }
        )
        assert response.status_code == 200

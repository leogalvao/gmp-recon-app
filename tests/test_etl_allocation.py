"""
Tests for ETL module allocation functions.
Tests Decimal-based currency parsing and Largest Remainder Method.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.modules.etl import (
    parse_money_to_cents,
    allocate_largest_remainder,
    allocate_east_west,
    cents_to_display
)


class TestParseMoneytoCents:
    """Tests for parse_money_to_cents with Decimal precision."""

    def test_integer_input(self):
        """Test integer input."""
        assert parse_money_to_cents(100) == 10000
        assert parse_money_to_cents(0) == 0
        assert parse_money_to_cents(-50) == -5000

    def test_float_input(self):
        """Test float input with proper Decimal handling."""
        assert parse_money_to_cents(100.50) == 10050
        assert parse_money_to_cents(100.99) == 10099
        assert parse_money_to_cents(0.01) == 1
        assert parse_money_to_cents(-25.75) == -2575

    def test_string_currency_format(self):
        """Test various string currency formats."""
        assert parse_money_to_cents("$1,234.56") == 123456
        assert parse_money_to_cents("1234.56") == 123456
        assert parse_money_to_cents("1,234") == 123400
        assert parse_money_to_cents("$1234") == 123400

    def test_negative_formats(self):
        """Test negative number formats."""
        assert parse_money_to_cents("-$500.00") == -50000
        assert parse_money_to_cents("($1,000.00)") == -100000  # Accounting notation
        assert parse_money_to_cents("-1234.56") == -123456

    def test_whitespace_handling(self):
        """Test whitespace handling."""
        assert parse_money_to_cents(" 715,643.50 ") == 71564350
        assert parse_money_to_cents("  $100  ") == 10000

    def test_empty_and_null(self):
        """Test empty and null inputs."""
        assert parse_money_to_cents(None) == 0
        assert parse_money_to_cents("") == 0
        assert parse_money_to_cents(" - ") == 0
        assert parse_money_to_cents("-") == 0

    def test_precision_no_float_drift(self):
        """Test that Decimal prevents float precision errors."""
        # This test catches float drift issues like 0.1 + 0.2 != 0.3
        assert parse_money_to_cents(0.1) == 10
        assert parse_money_to_cents(0.01) == 1
        assert parse_money_to_cents("715643.50") == 71564350

        # Test rounding with Decimal
        assert parse_money_to_cents(99.995) == 10000  # Round half up
        assert parse_money_to_cents(99.994) == 9999

    def test_malformed_input(self):
        """Test malformed inputs return 0."""
        assert parse_money_to_cents("abc") == 0
        assert parse_money_to_cents("$..50") == 0  # Multiple decimals


class TestAllocateLargestRemainder:
    """Tests for Largest Remainder Method (Hamilton's Method)."""

    def test_simple_allocation(self):
        """Test simple 50/50 allocation."""
        result = allocate_largest_remainder(10000, [0.5, 0.5])
        assert sum(result) == 10000
        assert result == [5000, 5000]

    def test_uneven_allocation(self):
        """Test uneven allocation maintains exact sum."""
        result = allocate_largest_remainder(10000, [0.333, 0.667])
        assert sum(result) == 10000
        # 3330 + 6670 = 10000

    def test_three_way_allocation(self):
        """Test three-way allocation."""
        result = allocate_largest_remainder(100, [0.33, 0.33, 0.34])
        assert sum(result) == 100
        assert len(result) == 3

    def test_edge_case_single_weight(self):
        """Test single weight gets all."""
        result = allocate_largest_remainder(5000, [1.0])
        assert result == [5000]

    def test_edge_case_empty_weights(self):
        """Test empty weights returns empty list."""
        result = allocate_largest_remainder(5000, [])
        assert result == []

    def test_zero_weights(self):
        """Test zero weights fallback to equal split."""
        result = allocate_largest_remainder(100, [0, 0])
        assert sum(result) == 100
        assert result[0] == 50

    def test_remainder_distribution(self):
        """Test that remainders are distributed correctly."""
        # 100 cents, [60%, 40%] -> 60 + 40 = 100
        result = allocate_largest_remainder(100, [0.6, 0.4])
        assert sum(result) == 100

        # 99 cents, [33.33%, 33.33%, 33.34%] -> remainders distributed
        result = allocate_largest_remainder(99, [0.333, 0.333, 0.334])
        assert sum(result) == 99

    def test_large_amounts(self):
        """Test with large amounts."""
        result = allocate_largest_remainder(123456789, [0.4, 0.35, 0.25])
        assert sum(result) == 123456789

    def test_penny_perfect(self):
        """Test penny-perfect allocation - no drift."""
        for total in [100, 1000, 10000, 99999, 123456]:
            for weights in [[0.5, 0.5], [0.333, 0.667], [0.25, 0.25, 0.5]]:
                result = allocate_largest_remainder(total, weights)
                assert sum(result) == total, f"Failed for {total} with {weights}"


class TestAllocateEastWest:
    """Tests for allocate_east_west convenience function."""

    def test_equal_split(self):
        """Test equal 50/50 split."""
        east, west = allocate_east_west(10000, 0.5, 0.5)
        assert east == 5000
        assert west == 5000
        assert east + west == 10000

    def test_uneven_split(self):
        """Test uneven split."""
        east, west = allocate_east_west(10000, 0.333, 0.667)
        assert east + west == 10000

    def test_all_east(self):
        """Test all to east."""
        east, west = allocate_east_west(5000, 1.0, 0.0)
        assert east == 5000
        assert west == 0

    def test_all_west(self):
        """Test all to west."""
        east, west = allocate_east_west(5000, 0.0, 1.0)
        assert east == 0
        assert west == 5000

    def test_exact_tie_out(self):
        """Test that east + west always equals total."""
        test_cases = [
            (10000, 0.333, 0.667),
            (12345, 0.45, 0.55),
            (99999, 0.123, 0.877),
            (1, 0.5, 0.5),
        ]
        for total, pct_e, pct_w in test_cases:
            east, west = allocate_east_west(total, pct_e, pct_w)
            assert east + west == total, f"Failed for {total}, {pct_e}, {pct_w}"


class TestCentsToDisplay:
    """Tests for cents to display string conversion."""

    def test_positive_amounts(self):
        """Test positive amounts."""
        assert cents_to_display(100) == "$1.00"
        assert cents_to_display(12345) == "$123.45"
        assert cents_to_display(1234567) == "$12,345.67"

    def test_negative_amounts(self):
        """Test negative amounts."""
        assert cents_to_display(-100) == "-$1.00"
        assert cents_to_display(-12345) == "-$123.45"

    def test_zero(self):
        """Test zero."""
        assert cents_to_display(0) == "$0.00"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

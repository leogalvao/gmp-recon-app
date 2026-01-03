"""
Tests for the configuration loader.
"""
import pytest
import tempfile
from pathlib import Path

from app.config import GMPConfig, get_config, reload_config, ConfigurationError


class TestGMPConfig:
    """Tests for GMPConfig class."""

    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = get_config()
        assert config.version == "1.0.0"
        assert len(config.gmp_divisions) == 31

    def test_get_division_name(self):
        """Test getting division name by key."""
        config = get_config()
        assert config.get_division_name("4") == "Concrete"
        assert config.get_division_name("26") == "Electrical & Fire Alarm"
        assert config.get_division_name("999") is None

    def test_get_division_aliases(self):
        """Test getting division aliases."""
        config = get_config()
        aliases = config.get_division_aliases("4")
        assert "Concrete Work" in aliases
        assert "Cast-in-Place" in aliases

    def test_get_all_division_names(self):
        """Test getting all division names."""
        config = get_config()
        names = config.get_all_division_names()
        assert len(names) == 31
        assert "Concrete" in names
        assert "Masonry" in names

    def test_find_division_by_name(self):
        """Test finding division key by name."""
        config = get_config()
        assert config.find_division_by_name("Concrete") == "4"
        assert config.find_division_by_name("concrete") == "4"  # case-insensitive
        assert config.find_division_by_name("Cast-in-Place") == "4"  # alias
        assert config.find_division_by_name("Unknown") is None


class TestFuzzyMatching:
    """Tests for fuzzy matching configuration."""

    def test_fuzzy_matching_enabled(self):
        """Test fuzzy matching enabled flag."""
        config = get_config()
        assert config.fuzzy_matching_enabled is True

    def test_fuzzy_algorithm(self):
        """Test fuzzy algorithm setting."""
        config = get_config()
        assert config.fuzzy_algorithm == "token_sort_ratio"

    def test_get_fuzzy_thresholds(self):
        """Test getting fuzzy thresholds by context."""
        config = get_config()

        budget_thresholds = config.get_fuzzy_thresholds("budget_to_gmp")
        assert budget_thresholds["min_confidence"] == 85
        assert budget_thresholds["auto_accept"] == 95

        direct_thresholds = config.get_fuzzy_thresholds("direct_to_budget")
        assert direct_thresholds["min_confidence"] == 60


class TestSuggestionEngine:
    """Tests for suggestion engine configuration."""

    def test_suggestion_weights(self):
        """Test suggestion weight values."""
        config = get_config()
        weights = config.suggestion_weights
        assert weights["code_prefix_match"] == 0.40
        assert weights["cost_type_match"] == 0.20
        assert weights["text_similarity"] == 0.30
        assert weights["historical_pattern"] == 0.10

    def test_weights_sum_to_one(self):
        """Test that suggestion weights sum to 1.0."""
        config = get_config()
        weights = config.suggestion_weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_confidence_bands(self):
        """Test confidence band thresholds."""
        config = get_config()
        assert config.get_confidence_band(0.90) == "high"
        assert config.get_confidence_band(0.85) == "high"
        assert config.get_confidence_band(0.70) == "medium"
        assert config.get_confidence_band(0.60) == "medium"
        assert config.get_confidence_band(0.30) == "low"
        assert config.get_confidence_band(0.0) == "low"

    def test_cost_type_scoring(self):
        """Test cost type compatibility scoring."""
        config = get_config()
        # Perfect matches
        assert config.get_cost_type_score("L", "L") == 1.0
        assert config.get_cost_type_score("M", "M") == 1.0
        assert config.get_cost_type_score("S", "S") == 1.0
        # Partial matches
        assert config.get_cost_type_score("L", "S") == 0.5
        assert config.get_cost_type_score("M", "S") == 0.4
        # Default for unknown
        assert config.get_cost_type_score("X", "Y") == 0.2


class TestAllocations:
    """Tests for allocation configuration."""

    def test_default_allocation(self):
        """Test default allocation values."""
        config = get_config()
        default = config.default_allocation
        assert default["west"] == 0.50
        assert default["east"] == 0.50

    def test_division_allocation(self):
        """Test getting division-specific allocation."""
        config = get_config()
        alloc = config.get_division_allocation("1")
        assert "west" in alloc
        assert "east" in alloc

    def test_allocation_sum_tolerance(self):
        """Test allocation sum tolerance."""
        config = get_config()
        assert config.allocation_sum_tolerance == 0.001


class TestDuplicateDetection:
    """Tests for duplicate detection configuration."""

    def test_duplicate_detection_enabled(self):
        """Test duplicate detection enabled flag."""
        config = get_config()
        assert config.duplicate_detection_enabled is True

    def test_duplicate_method_config(self):
        """Test getting duplicate method configuration."""
        config = get_config()
        exact = config.get_duplicate_method_config("exact")
        assert exact["enabled"] is True
        assert exact["auto_exclude"] is True


class TestMLForecasting:
    """Tests for ML forecasting configuration."""

    def test_ml_forecasting_enabled(self):
        """Test ML forecasting enabled flag."""
        config = get_config()
        assert config.ml_forecasting_enabled is True

    def test_model_config(self):
        """Test getting model configuration."""
        config = get_config()
        lr_config = config.get_model_config("linear_regression")
        assert lr_config["enabled"] is True
        assert lr_config["min_data_points"] == 5


class TestDataValidation:
    """Tests for data validation configuration."""

    def test_required_columns(self):
        """Test getting required columns by file type."""
        config = get_config()
        gmp_cols = config.get_required_columns("gmp")
        assert "GMP" in gmp_cols
        assert "Amount Total" in gmp_cols

        budget_cols = config.get_required_columns("budget")
        assert "Budget Code" in budget_cols

    def test_validation_warnings(self):
        """Test validation warning thresholds."""
        config = get_config()
        warnings = config.validation_warnings
        assert warnings["missing_vendor_pct"] == 0.05


class TestUIConfig:
    """Tests for UI configuration."""

    def test_currency_config(self):
        """Test currency formatting configuration."""
        config = get_config()
        currency = config.currency_config
        assert currency["symbol"] == "$"
        assert currency["decimal_places"] == 2

    def test_table_config(self):
        """Test table configuration."""
        config = get_config()
        tables = config.table_config
        assert tables["default_page_size"] == 50


class TestAuditConfig:
    """Tests for audit configuration."""

    def test_audit_retention(self):
        """Test audit retention days."""
        config = get_config()
        assert config.audit_retention_days == 365


class TestRawAccess:
    """Tests for raw configuration access."""

    def test_get_method(self):
        """Test get method with default."""
        config = get_config()
        assert config.get("version") == "1.0.0"
        assert config.get("nonexistent", "default") == "default"

    def test_getitem(self):
        """Test dictionary-style access."""
        config = get_config()
        assert config["version"] == "1.0.0"

    def test_contains(self):
        """Test key existence check."""
        config = get_config()
        assert "gmp_divisions" in config
        assert "nonexistent" not in config


class TestConfigurationError:
    """Tests for configuration error handling."""

    def test_missing_file(self):
        """Test error on missing config file."""
        with pytest.raises(ConfigurationError) as exc_info:
            GMPConfig(Path("/nonexistent/path.yaml"))
        assert "not found" in str(exc_info.value)

    def test_invalid_yaml(self):
        """Test error on invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                GMPConfig(temp_path)
            assert "Invalid YAML" in str(exc_info.value)
        finally:
            temp_path.unlink()

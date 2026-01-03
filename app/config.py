"""
Configuration loader for GMP Reconciliation App.

Loads settings from gmp_mapping_config.yaml and provides typed access
to all configuration sections.
"""
import os
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

import yaml


# Default config path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "gmp_mapping_config.yaml"


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class GMPConfig:
    """
    Configuration manager for GMP Reconciliation App.

    Loads YAML configuration and provides typed access to all sections.
    Use get_config() to obtain the singleton instance.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self._config_path.exists():
            raise ConfigurationError(f"Config file not found: {self._config_path}")

        try:
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")

        if not isinstance(self._config, dict):
            raise ConfigurationError("Config file must contain a YAML mapping")

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load()
        # Clear the cached singleton to force reload on next get_config()
        get_config.cache_clear()

    @property
    def version(self) -> str:
        """Configuration file version."""
        return self._config.get("version", "unknown")

    # =========================================================================
    # GMP Divisions
    # =========================================================================

    @property
    def gmp_divisions(self) -> dict:
        """All GMP division definitions."""
        return self._config.get("gmp_divisions", {})

    def get_division_name(self, division_key: str) -> Optional[str]:
        """Get GMP division name by key (e.g., '4' -> 'Concrete')."""
        division = self.gmp_divisions.get(str(division_key))
        return division.get("name") if division else None

    def get_division_aliases(self, division_key: str) -> list[str]:
        """Get list of aliases for a division."""
        division = self.gmp_divisions.get(str(division_key))
        return division.get("aliases", []) if division else []

    def get_all_division_names(self) -> list[str]:
        """Get list of all GMP division names."""
        return [d.get("name") for d in self.gmp_divisions.values() if d.get("name")]

    def find_division_by_name(self, name: str) -> Optional[str]:
        """Find division key by name or alias (case-insensitive)."""
        name_lower = name.lower().strip()
        for key, division in self.gmp_divisions.items():
            if division.get("name", "").lower().strip() == name_lower:
                return key
            for alias in division.get("aliases", []):
                if alias.lower() == name_lower:
                    return key
        return None

    # =========================================================================
    # Fuzzy Matching
    # =========================================================================

    @property
    def fuzzy_matching(self) -> dict:
        """Fuzzy matching configuration."""
        return self._config.get("fuzzy_matching", {})

    @property
    def fuzzy_matching_enabled(self) -> bool:
        """Whether fuzzy matching is enabled."""
        return self.fuzzy_matching.get("enabled", True)

    @property
    def fuzzy_algorithm(self) -> str:
        """Fuzzy matching algorithm to use."""
        return self.fuzzy_matching.get("algorithm", "token_sort_ratio")

    def get_fuzzy_thresholds(self, context: str) -> dict:
        """
        Get fuzzy matching thresholds for a context.

        Args:
            context: One of 'budget_to_gmp', 'tier2_to_gmp', 'direct_to_budget'

        Returns:
            Dict with min_confidence, auto_accept, review_required thresholds
        """
        thresholds = self.fuzzy_matching.get("thresholds", {})
        return thresholds.get(context, {
            "min_confidence": 70,
            "auto_accept": 90,
            "review_required": 50
        })

    # =========================================================================
    # Suggestion Engine
    # =========================================================================

    @property
    def suggestion_engine(self) -> dict:
        """Suggestion engine configuration."""
        return self._config.get("suggestion_engine", {})

    @property
    def suggestion_weights(self) -> dict:
        """Scoring weights for suggestion engine."""
        return self.suggestion_engine.get("weights", {
            "code_prefix_match": 0.40,
            "cost_type_match": 0.20,
            "text_similarity": 0.30,
            "historical_pattern": 0.10
        })

    @property
    def confidence_bands(self) -> dict:
        """Confidence band definitions."""
        return self.suggestion_engine.get("confidence_bands", {})

    def get_confidence_band(self, score: float) -> str:
        """
        Get confidence band name for a score.

        Args:
            score: Score between 0.0 and 1.0

        Returns:
            Band name: 'high', 'medium', or 'low'
        """
        bands = self.confidence_bands
        if score >= bands.get("high", {}).get("min_score", 0.85):
            return "high"
        elif score >= bands.get("medium", {}).get("min_score", 0.60):
            return "medium"
        else:
            return "low"

    @property
    def cost_type_compatibility(self) -> list:
        """Cost type compatibility matrix entries."""
        return self.suggestion_engine.get("cost_type_compatibility", [])

    @property
    def cost_type_default_score(self) -> float:
        """Default score for unmapped cost type combinations."""
        return self.suggestion_engine.get("default_score", 0.2)

    def get_cost_type_score(self, source_type: str, target_type: str) -> float:
        """
        Get compatibility score for a cost type pair.

        Args:
            source_type: Source cost type (e.g., 'L', 'M', 'S', 'O')
            target_type: Target cost type

        Returns:
            Compatibility score between 0.0 and 1.0
        """
        for entry in self.cost_type_compatibility:
            types = entry.get("types", [])
            if len(types) == 2 and types[0] == source_type and types[1] == target_type:
                return entry.get("score", self.cost_type_default_score)
        return self.cost_type_default_score

    # =========================================================================
    # Allocations
    # =========================================================================

    @property
    def allocations(self) -> dict:
        """Allocation configuration."""
        return self._config.get("allocations", {})

    @property
    def default_allocation(self) -> dict:
        """Default West/East allocation split."""
        return self.allocations.get("default", {"west": 0.50, "east": 0.50})

    def get_division_allocation(self, division_key: str) -> dict:
        """
        Get default allocation for a division.

        Args:
            division_key: Division key (e.g., '4' for Concrete)

        Returns:
            Dict with 'west' and 'east' percentages
        """
        division_defaults = self.allocations.get("division_defaults", {})
        return division_defaults.get(str(division_key), self.default_allocation)

    @property
    def allocation_sum_tolerance(self) -> float:
        """Tolerance for west + east != 1.0 validation."""
        validation = self.allocations.get("validation", {})
        return validation.get("sum_tolerance", 0.001)

    # =========================================================================
    # Duplicate Detection
    # =========================================================================

    @property
    def duplicate_detection(self) -> dict:
        """Duplicate detection configuration."""
        return self._config.get("duplicate_detection", {})

    @property
    def duplicate_detection_enabled(self) -> bool:
        """Whether duplicate detection is enabled."""
        return self.duplicate_detection.get("enabled", True)

    def get_duplicate_method_config(self, method: str) -> dict:
        """
        Get configuration for a duplicate detection method.

        Args:
            method: One of 'exact', 'fuzzy', 'reversal'
        """
        methods = self.duplicate_detection.get("methods", {})
        return methods.get(method, {})

    # =========================================================================
    # ML Forecasting
    # =========================================================================

    @property
    def ml_forecasting(self) -> dict:
        """ML forecasting configuration."""
        return self._config.get("ml_forecasting", {})

    @property
    def ml_forecasting_enabled(self) -> bool:
        """Whether ML forecasting is enabled."""
        return self.ml_forecasting.get("enabled", True)

    def get_model_config(self, model_name: str) -> dict:
        """
        Get configuration for an ML model.

        Args:
            model_name: One of 'linear_regression', 'mlp'
        """
        models = self.ml_forecasting.get("models", {})
        return models.get(model_name, {})

    # =========================================================================
    # Data Validation
    # =========================================================================

    @property
    def data_validation(self) -> dict:
        """Data validation configuration."""
        return self._config.get("data_validation", {})

    def get_required_columns(self, file_type: str) -> list[str]:
        """
        Get required columns for a file type.

        Args:
            file_type: One of 'gmp', 'budget', 'direct_costs'
        """
        required = self.data_validation.get("required_columns", {})
        return required.get(file_type, [])

    @property
    def validation_warnings(self) -> dict:
        """Warning thresholds for data validation."""
        return self.data_validation.get("warnings", {})

    # =========================================================================
    # UI Configuration
    # =========================================================================

    @property
    def ui(self) -> dict:
        """UI configuration."""
        return self._config.get("ui", {})

    @property
    def currency_config(self) -> dict:
        """Currency formatting configuration."""
        return self.ui.get("currency", {
            "symbol": "$",
            "decimal_places": 2,
            "thousands_separator": ","
        })

    @property
    def table_config(self) -> dict:
        """Table display configuration."""
        return self.ui.get("tables", {})

    # =========================================================================
    # Audit Configuration
    # =========================================================================

    @property
    def audit(self) -> dict:
        """Audit trail configuration."""
        return self._config.get("audit", {})

    @property
    def audit_retention_days(self) -> int:
        """Number of days to retain audit records."""
        return self.audit.get("retention_days", 365)

    # =========================================================================
    # Integrations
    # =========================================================================

    @property
    def integrations(self) -> dict:
        """Integration settings."""
        return self._config.get("integrations", {})

    @property
    def file_watcher_config(self) -> dict:
        """File watcher configuration."""
        return self.integrations.get("file_watcher", {})

    # =========================================================================
    # Raw Access
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value by key."""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> GMPConfig:
    """
    Get the singleton configuration instance.

    Args:
        config_path: Optional path to config file. Only used on first call.

    Returns:
        GMPConfig singleton instance
    """
    path = Path(config_path) if config_path else None
    return GMPConfig(path)


def reload_config() -> GMPConfig:
    """Reload configuration from disk and return new instance."""
    get_config.cache_clear()
    return get_config()


# Build division key -> name lookup for fast access
def build_division_lookup() -> dict[str, str]:
    """Build a lookup dict mapping division keys to names."""
    config = get_config()
    return {key: div.get("name", "") for key, div in config.gmp_divisions.items()}

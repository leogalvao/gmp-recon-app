"""
Feature Flags - Gradual rollout system for new features.

Provides:
1. Per-project feature flag management
2. Percentage-based rollout
3. Override capabilities for testing
4. Audit logging of flag changes
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)


class RolloutStrategy(str, Enum):
    """Rollout strategies for feature flags."""
    DISABLED = "disabled"          # Flag is off for everyone
    ENABLED = "enabled"            # Flag is on for everyone
    PERCENTAGE = "percentage"      # Flag is on for X% of entities
    ALLOWLIST = "allowlist"        # Flag is on only for specific entities
    DENYLIST = "denylist"          # Flag is on for everyone except specific entities


@dataclass
class FeatureFlagConfig:
    """Configuration for a feature flag."""
    name: str
    description: str
    strategy: RolloutStrategy = RolloutStrategy.DISABLED
    percentage: float = 0.0  # For percentage rollout (0-100)
    allowlist: Set[int] = None  # Entity IDs where flag is enabled
    denylist: Set[int] = None   # Entity IDs where flag is disabled
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.allowlist is None:
            self.allowlist = set()
        if self.denylist is None:
            self.denylist = set()
        if self.metadata is None:
            self.metadata = {}


class FeatureFlag:
    """
    Feature flag for gradual rollout of new functionality.

    Example usage:
        multi_project_flag = FeatureFlag('multi_project_forecasting')

        if multi_project_flag.is_enabled(project_id=123):
            # Use new multi-project forecasting
            return get_forecasts_v2(project_id)
        else:
            # Use legacy forecasting
            return get_forecasts_legacy(project_id)
    """

    # In-memory flag registry (for simplicity; could use Redis/DB in production)
    _registry: Dict[str, FeatureFlagConfig] = {}

    def __init__(self, name: str, config: Optional[FeatureFlagConfig] = None):
        self.name = name

        if name not in self._registry:
            # Create default config if not exists
            self._registry[name] = config or FeatureFlagConfig(
                name=name,
                description=f"Feature flag: {name}",
            )

    @property
    def config(self) -> FeatureFlagConfig:
        return self._registry[self.name]

    def is_enabled(self, entity_id: Optional[int] = None) -> bool:
        """
        Check if the feature is enabled.

        Args:
            entity_id: Optional entity ID (project_id, user_id, etc.)

        Returns:
            True if feature is enabled for this entity
        """
        config = self.config

        if config.strategy == RolloutStrategy.DISABLED:
            return False

        if config.strategy == RolloutStrategy.ENABLED:
            return True

        if entity_id is None:
            # No entity specified, use default behavior
            return config.strategy == RolloutStrategy.ENABLED

        if config.strategy == RolloutStrategy.ALLOWLIST:
            return entity_id in config.allowlist

        if config.strategy == RolloutStrategy.DENYLIST:
            return entity_id not in config.denylist

        if config.strategy == RolloutStrategy.PERCENTAGE:
            # Deterministic hash-based percentage check
            hash_value = hash(f"{self.name}:{entity_id}") % 100
            return hash_value < config.percentage

        return False

    def enable(self, entity_id: Optional[int] = None):
        """
        Enable the feature flag.

        Args:
            entity_id: If provided, add to allowlist; otherwise enable globally
        """
        if entity_id is not None:
            self.config.allowlist.add(entity_id)
            if self.config.strategy == RolloutStrategy.DISABLED:
                self.config.strategy = RolloutStrategy.ALLOWLIST
            logger.info(f"Feature '{self.name}' enabled for entity {entity_id}")
        else:
            self.config.strategy = RolloutStrategy.ENABLED
            logger.info(f"Feature '{self.name}' enabled globally")

    def disable(self, entity_id: Optional[int] = None):
        """
        Disable the feature flag.

        Args:
            entity_id: If provided, remove from allowlist; otherwise disable globally
        """
        if entity_id is not None:
            self.config.allowlist.discard(entity_id)
            self.config.denylist.add(entity_id)
            logger.info(f"Feature '{self.name}' disabled for entity {entity_id}")
        else:
            self.config.strategy = RolloutStrategy.DISABLED
            logger.info(f"Feature '{self.name}' disabled globally")

    def set_percentage(self, percentage: float):
        """
        Set percentage-based rollout.

        Args:
            percentage: Percentage of entities to enable (0-100)
        """
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")

        self.config.strategy = RolloutStrategy.PERCENTAGE
        self.config.percentage = percentage
        logger.info(f"Feature '{self.name}' set to {percentage}% rollout")

    def get_status(self) -> Dict[str, Any]:
        """Get current flag status."""
        return {
            'name': self.name,
            'strategy': self.config.strategy.value,
            'percentage': self.config.percentage,
            'allowlist_count': len(self.config.allowlist),
            'denylist_count': len(self.config.denylist),
            'metadata': self.config.metadata,
        }

    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """List all registered feature flags."""
        return [
            FeatureFlag(name).get_status()
            for name in cls._registry.keys()
        ]

    @classmethod
    def reset(cls, name: Optional[str] = None):
        """Reset feature flags (for testing)."""
        if name:
            if name in cls._registry:
                del cls._registry[name]
        else:
            cls._registry.clear()


# Pre-defined feature flags for the multi-project system
class FeatureFlags:
    """
    Registry of feature flags for the application.
    """

    # Multi-project forecasting system
    MULTI_PROJECT_FORECASTING = FeatureFlag(
        'multi_project_forecasting',
        FeatureFlagConfig(
            name='multi_project_forecasting',
            description='Enable new multi-project ML forecasting system',
            strategy=RolloutStrategy.DISABLED,
        )
    )

    # Canonical trade mapping
    CANONICAL_TRADE_MAPPING = FeatureFlag(
        'canonical_trade_mapping',
        FeatureFlagConfig(
            name='canonical_trade_mapping',
            description='Enable canonical trade taxonomy mapping',
            strategy=RolloutStrategy.DISABLED,
        )
    )

    # Feature store for ML
    FEATURE_STORE = FeatureFlag(
        'feature_store',
        FeatureFlagConfig(
            name='feature_store',
            description='Enable canonical cost feature store',
            strategy=RolloutStrategy.DISABLED,
        )
    )

    # Probabilistic forecasting output
    PROBABILISTIC_FORECASTS = FeatureFlag(
        'probabilistic_forecasts',
        FeatureFlagConfig(
            name='probabilistic_forecasts',
            description='Enable uncertainty quantification in forecasts',
            strategy=RolloutStrategy.DISABLED,
        )
    )

    @classmethod
    def enable_for_project(cls, project_id: int):
        """Enable all multi-project features for a specific project."""
        cls.MULTI_PROJECT_FORECASTING.enable(project_id)
        cls.CANONICAL_TRADE_MAPPING.enable(project_id)
        cls.FEATURE_STORE.enable(project_id)
        cls.PROBABILISTIC_FORECASTS.enable(project_id)
        logger.info(f"All multi-project features enabled for project {project_id}")

    @classmethod
    def disable_for_project(cls, project_id: int):
        """Disable all multi-project features for a specific project."""
        cls.MULTI_PROJECT_FORECASTING.disable(project_id)
        cls.CANONICAL_TRADE_MAPPING.disable(project_id)
        cls.FEATURE_STORE.disable(project_id)
        cls.PROBABILISTIC_FORECASTS.disable(project_id)
        logger.info(f"All multi-project features disabled for project {project_id}")

    @classmethod
    def enable_globally(cls):
        """Enable all multi-project features globally."""
        cls.MULTI_PROJECT_FORECASTING.enable()
        cls.CANONICAL_TRADE_MAPPING.enable()
        cls.FEATURE_STORE.enable()
        cls.PROBABILISTIC_FORECASTS.enable()
        logger.info("All multi-project features enabled globally")

    @classmethod
    def set_rollout_percentage(cls, percentage: float):
        """Set percentage-based rollout for all features."""
        cls.MULTI_PROJECT_FORECASTING.set_percentage(percentage)
        cls.CANONICAL_TRADE_MAPPING.set_percentage(percentage)
        cls.FEATURE_STORE.set_percentage(percentage)
        cls.PROBABILISTIC_FORECASTS.set_percentage(percentage)
        logger.info(f"All multi-project features set to {percentage}% rollout")


def get_feature_flag(name: str) -> FeatureFlag:
    """Get a feature flag by name."""
    return FeatureFlag(name)

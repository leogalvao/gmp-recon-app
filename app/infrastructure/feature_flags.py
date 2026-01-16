"""
Feature Flags - Gradual rollout system for new features.

Provides:
1. Per-project feature flag management
2. Percentage-based rollout
3. Database persistence for flag states
4. Override capabilities for testing
5. Audit logging of flag changes
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

from sqlalchemy.orm import Session

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
    allowlist: Set[int] = field(default_factory=set)  # Entity IDs where flag is enabled
    denylist: Set[int] = field(default_factory=set)   # Entity IDs where flag is disabled
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureFlagStore:
    """
    Database-backed storage for feature flags.
    Provides persistence across process restarts.
    """

    _db_getter: Optional[Callable[[], Session]] = None

    @classmethod
    def set_db_getter(cls, db_getter: Callable[[], Session]):
        """Set the database session getter function."""
        cls._db_getter = db_getter

    @classmethod
    def _get_db(cls) -> Optional[Session]:
        """Get a database session."""
        if cls._db_getter:
            return cls._db_getter()
        return None

    @classmethod
    def load_flag_state(cls, flag_name: str) -> Optional[Dict[str, Any]]:
        """Load flag state from database."""
        db = cls._get_db()
        if not db:
            return None

        try:
            from app.models import FeatureFlagState, FeatureFlagEntity

            # Get flag state
            state = db.query(FeatureFlagState).filter(
                FeatureFlagState.flag_name == flag_name
            ).first()

            if not state:
                return None

            # Get entity entries (allowlist/denylist)
            entities = db.query(FeatureFlagEntity).filter(
                FeatureFlagEntity.flag_name == flag_name
            ).all()

            allowlist = {e.entity_id for e in entities if e.is_enabled}
            denylist = {e.entity_id for e in entities if not e.is_enabled}

            return {
                'strategy': state.strategy,
                'percentage': state.percentage,
                'allowlist': allowlist,
                'denylist': denylist,
            }
        except Exception as e:
            logger.warning(f"Failed to load flag state from DB: {e}")
            return None
        finally:
            db.close()

    @classmethod
    def save_flag_state(
        cls,
        flag_name: str,
        strategy: str,
        percentage: float = 0.0,
        allowlist: Optional[Set[int]] = None,
        denylist: Optional[Set[int]] = None,
    ):
        """Save flag state to database."""
        db = cls._get_db()
        if not db:
            return

        try:
            from app.models import FeatureFlagState, FeatureFlagEntity

            # Upsert flag state
            state = db.query(FeatureFlagState).filter(
                FeatureFlagState.flag_name == flag_name
            ).first()

            if state:
                state.strategy = strategy
                state.percentage = percentage
                state.updated_at = datetime.utcnow()
            else:
                state = FeatureFlagState(
                    flag_name=flag_name,
                    strategy=strategy,
                    percentage=percentage,
                )
                db.add(state)

            # Update entity entries
            if allowlist is not None:
                for entity_id in allowlist:
                    existing = db.query(FeatureFlagEntity).filter(
                        FeatureFlagEntity.flag_name == flag_name,
                        FeatureFlagEntity.entity_id == entity_id,
                    ).first()

                    if existing:
                        existing.is_enabled = True
                    else:
                        db.add(FeatureFlagEntity(
                            flag_name=flag_name,
                            entity_id=entity_id,
                            is_enabled=True,
                        ))

            if denylist is not None:
                for entity_id in denylist:
                    existing = db.query(FeatureFlagEntity).filter(
                        FeatureFlagEntity.flag_name == flag_name,
                        FeatureFlagEntity.entity_id == entity_id,
                    ).first()

                    if existing:
                        existing.is_enabled = False
                    else:
                        db.add(FeatureFlagEntity(
                            flag_name=flag_name,
                            entity_id=entity_id,
                            is_enabled=False,
                        ))

            db.commit()
            logger.debug(f"Saved flag state for '{flag_name}' to database")
        except Exception as e:
            logger.warning(f"Failed to save flag state to DB: {e}")
            db.rollback()
        finally:
            db.close()

    @classmethod
    def add_entity(cls, flag_name: str, entity_id: int, is_enabled: bool):
        """Add or update a single entity entry."""
        db = cls._get_db()
        if not db:
            return

        try:
            from app.models import FeatureFlagEntity

            existing = db.query(FeatureFlagEntity).filter(
                FeatureFlagEntity.flag_name == flag_name,
                FeatureFlagEntity.entity_id == entity_id,
            ).first()

            if existing:
                existing.is_enabled = is_enabled
            else:
                db.add(FeatureFlagEntity(
                    flag_name=flag_name,
                    entity_id=entity_id,
                    is_enabled=is_enabled,
                ))

            db.commit()
        except Exception as e:
            logger.warning(f"Failed to add entity to DB: {e}")
            db.rollback()
        finally:
            db.close()


class FeatureFlag:
    """
    Feature flag for gradual rollout of new functionality.

    Supports both in-memory and database-backed persistence.
    Database persistence is used when available, with in-memory
    fallback for testing or when DB is not configured.

    Example usage:
        multi_project_flag = FeatureFlag('multi_project_forecasting')

        if multi_project_flag.is_enabled(project_id=123):
            # Use new multi-project forecasting
            return get_forecasts_v2(project_id)
        else:
            # Use legacy forecasting
            return get_forecasts_legacy(project_id)
    """

    # In-memory flag registry (cache + fallback when DB not available)
    _registry: Dict[str, FeatureFlagConfig] = {}

    def __init__(self, name: str, config: Optional[FeatureFlagConfig] = None):
        self.name = name
        self._default_config = config

        if name not in self._registry:
            # Try to load from database first
            db_state = FeatureFlagStore.load_flag_state(name)

            if db_state:
                # Use database state
                self._registry[name] = FeatureFlagConfig(
                    name=name,
                    description=config.description if config else f"Feature flag: {name}",
                    strategy=RolloutStrategy(db_state['strategy']),
                    percentage=db_state['percentage'],
                    allowlist=db_state['allowlist'],
                    denylist=db_state['denylist'],
                )
            else:
                # Use provided config or default
                self._registry[name] = config or FeatureFlagConfig(
                    name=name,
                    description=f"Feature flag: {name}",
                )

    @property
    def config(self) -> FeatureFlagConfig:
        # Re-create config if registry was cleared (e.g., by reset())
        if self.name not in self._registry:
            # Try DB first
            db_state = FeatureFlagStore.load_flag_state(self.name)
            if db_state:
                self._registry[self.name] = FeatureFlagConfig(
                    name=self.name,
                    description=self._default_config.description if self._default_config else f"Feature flag: {self.name}",
                    strategy=RolloutStrategy(db_state['strategy']),
                    percentage=db_state['percentage'],
                    allowlist=db_state['allowlist'],
                    denylist=db_state['denylist'],
                )
            else:
                self._registry[self.name] = self._default_config or FeatureFlagConfig(
                    name=self.name,
                    description=f"Feature flag: {self.name}",
                )
        return self._registry[self.name]

    def _sync_from_db(self):
        """Refresh config from database."""
        db_state = FeatureFlagStore.load_flag_state(self.name)
        if db_state:
            self.config.strategy = RolloutStrategy(db_state['strategy'])
            self.config.percentage = db_state['percentage']
            self.config.allowlist = db_state['allowlist']
            self.config.denylist = db_state['denylist']

    def _persist(self):
        """Persist current config to database."""
        FeatureFlagStore.save_flag_state(
            flag_name=self.name,
            strategy=self.config.strategy.value,
            percentage=self.config.percentage,
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
        )

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
            self.config.denylist.discard(entity_id)  # Remove from denylist if present
            if self.config.strategy == RolloutStrategy.DISABLED:
                self.config.strategy = RolloutStrategy.ALLOWLIST
            # Persist entity change and strategy
            FeatureFlagStore.add_entity(self.name, entity_id, is_enabled=True)
            self._persist()  # Also persist strategy change
            logger.info(f"Feature '{self.name}' enabled for entity {entity_id}")
        else:
            self.config.strategy = RolloutStrategy.ENABLED
            self._persist()
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
            # Persist entity change
            FeatureFlagStore.add_entity(self.name, entity_id, is_enabled=False)
            self._persist()  # Persist denylist change
            logger.info(f"Feature '{self.name}' disabled for entity {entity_id}")
        else:
            self.config.strategy = RolloutStrategy.DISABLED
            self._persist()
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
        self._persist()
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
        """Reset feature flags (for testing - clears in-memory cache only)."""
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
    def disable_globally(cls):
        """Disable all multi-project features globally."""
        cls.MULTI_PROJECT_FORECASTING.disable()
        cls.CANONICAL_TRADE_MAPPING.disable()
        cls.FEATURE_STORE.disable()
        cls.PROBABILISTIC_FORECASTS.disable()
        logger.info("All multi-project features disabled globally")

    @classmethod
    def set_rollout_percentage(cls, percentage: float):
        """Set percentage-based rollout for all features."""
        cls.MULTI_PROJECT_FORECASTING.set_percentage(percentage)
        cls.CANONICAL_TRADE_MAPPING.set_percentage(percentage)
        cls.FEATURE_STORE.set_percentage(percentage)
        cls.PROBABILISTIC_FORECASTS.set_percentage(percentage)
        logger.info(f"All multi-project features set to {percentage}% rollout")

    @classmethod
    def sync_from_db(cls):
        """Reload all flags from database."""
        cls.MULTI_PROJECT_FORECASTING._sync_from_db()
        cls.CANONICAL_TRADE_MAPPING._sync_from_db()
        cls.FEATURE_STORE._sync_from_db()
        cls.PROBABILISTIC_FORECASTS._sync_from_db()
        logger.info("Feature flags synced from database")


def get_feature_flag(name: str) -> FeatureFlag:
    """Get a feature flag by name."""
    return FeatureFlag(name)


def init_feature_flags():
    """Initialize feature flag system with database connection."""
    from app.models import get_db, SessionLocal

    def db_getter():
        return SessionLocal()

    FeatureFlagStore.set_db_getter(db_getter)
    logger.info("Feature flag database persistence initialized")

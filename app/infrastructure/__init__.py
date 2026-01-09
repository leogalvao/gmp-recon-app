"""
Infrastructure Layer - ML training, data processing, and repository implementations.

This module provides:
- Feature engineering for ML models
- Training pipeline orchestration
- Model registry and versioning
- Repository pattern for data access

Note: ML classes (FeatureEngineer, TrainingPipeline) use lazy loading to avoid
importing TensorFlow/PyTorch when only repository functionality is needed.
"""

from .repositories import (
    BaseRepository,
    GMPRepository,
    BudgetRepository,
    DirectCostRepository,
    ScheduleRepository,
)

# Lazy imports for ML classes to avoid TensorFlow dependency at import time
_FeatureEngineer = None
_TrainingPipeline = None
_TrainingConfig = None


def get_feature_engineer():
    """Get FeatureEngineer class (lazy import to avoid TensorFlow dependency)."""
    global _FeatureEngineer
    if _FeatureEngineer is None:
        from .ml.feature_engineering import FeatureEngineer
        _FeatureEngineer = FeatureEngineer
    return _FeatureEngineer


def get_training_pipeline():
    """Get TrainingPipeline class (lazy import to avoid TensorFlow dependency)."""
    global _TrainingPipeline
    if _TrainingPipeline is None:
        from .ml.training_pipeline import TrainingPipeline
        _TrainingPipeline = TrainingPipeline
    return _TrainingPipeline


def get_training_config():
    """Get TrainingConfig class (lazy import to avoid TensorFlow dependency)."""
    global _TrainingConfig
    if _TrainingConfig is None:
        from .ml.training_pipeline import TrainingConfig
        _TrainingConfig = TrainingConfig
    return _TrainingConfig


__all__ = [
    'get_feature_engineer',
    'get_training_pipeline',
    'get_training_config',
    'BaseRepository',
    'GMPRepository',
    'BudgetRepository',
    'DirectCostRepository',
    'ScheduleRepository',
]

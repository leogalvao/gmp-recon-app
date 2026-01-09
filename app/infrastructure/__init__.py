"""
Infrastructure Layer - ML training, data processing, and repository implementations.

This module provides:
- Feature engineering for ML models
- Training pipeline orchestration
- Model registry and versioning
- Repository pattern for data access
"""

# Repository imports (no external dependencies)
from .repositories import (
    BaseRepository,
    GMPRepository,
    BudgetRepository,
    DirectCostRepository,
    ScheduleRepository,
)

# Lazy imports for ML modules (require TensorFlow/PyTorch)
# These are imported on demand to avoid requiring ML dependencies for basic usage
def get_feature_engineer():
    """Lazy import of FeatureEngineer to avoid TensorFlow dependency."""
    from .ml.feature_engineering import FeatureEngineer
    return FeatureEngineer

def get_training_pipeline():
    """Lazy import of TrainingPipeline to avoid TensorFlow dependency."""
    from .ml.training_pipeline import TrainingPipeline, TrainingConfig
    return TrainingPipeline, TrainingConfig

__all__ = [
    'BaseRepository',
    'GMPRepository',
    'BudgetRepository',
    'DirectCostRepository',
    'ScheduleRepository',
    'get_feature_engineer',
    'get_training_pipeline',
]

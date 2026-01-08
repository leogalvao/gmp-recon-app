"""
Infrastructure Layer - ML training and data processing utilities.

This module provides:
- Feature engineering for ML models
- Training pipeline orchestration
- Model registry and versioning
"""

from .ml.feature_engineering import FeatureEngineer
from .ml.training_pipeline import TrainingPipeline, TrainingConfig

__all__ = ['FeatureEngineer', 'TrainingPipeline', 'TrainingConfig']

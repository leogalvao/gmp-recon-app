"""
Machine Learning Infrastructure - Training and feature processing.

Note: Uses lazy imports to avoid loading TensorFlow/PyTorch at import time.
Use get_* functions to access ML classes.
"""

# Lazy imports to avoid TensorFlow dependency at import time
_FeatureEngineer = None
_TrainingPipeline = None
_TrainingConfig = None


def get_feature_engineer():
    """Get FeatureEngineer class (lazy import)."""
    global _FeatureEngineer
    if _FeatureEngineer is None:
        from .feature_engineering import FeatureEngineer
        _FeatureEngineer = FeatureEngineer
    return _FeatureEngineer


def get_training_pipeline():
    """Get TrainingPipeline class (lazy import)."""
    global _TrainingPipeline
    if _TrainingPipeline is None:
        from .training_pipeline import TrainingPipeline
        _TrainingPipeline = TrainingPipeline
    return _TrainingPipeline


def get_training_config():
    """Get TrainingConfig class (lazy import)."""
    global _TrainingConfig
    if _TrainingConfig is None:
        from .training_pipeline import TrainingConfig
        _TrainingConfig = TrainingConfig
    return _TrainingConfig


__all__ = ['get_feature_engineer', 'get_training_pipeline', 'get_training_config']

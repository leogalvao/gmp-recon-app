"""
Forecasting Models - Neural network architectures for cost prediction.

Note: LSTM and Transformer models use lazy imports to avoid loading TensorFlow
at import time. Use get_* functions to access these classes.
"""

from .base_model import BaseForecaster, BuildingFeatures, ForecastResult

# Lazy imports for TensorFlow-dependent classes
_LSTMForecaster = None
_TransformerForecaster = None
_MultiProjectForecaster = None


def get_lstm_forecaster():
    """Get LSTMForecaster class (lazy import to avoid TensorFlow dependency)."""
    global _LSTMForecaster
    if _LSTMForecaster is None:
        from .lstm_forecaster import LSTMForecaster
        _LSTMForecaster = LSTMForecaster
    return _LSTMForecaster


def get_transformer_forecaster():
    """Get TransformerForecaster class (lazy import to avoid TensorFlow dependency)."""
    global _TransformerForecaster
    if _TransformerForecaster is None:
        from .transformer_forecaster import TransformerForecaster
        _TransformerForecaster = TransformerForecaster
    return _TransformerForecaster


def get_multi_project_forecaster():
    """Get MultiProjectForecaster class (lazy import to avoid TensorFlow dependency)."""
    global _MultiProjectForecaster
    if _MultiProjectForecaster is None:
        from .multi_project_forecaster import MultiProjectForecaster
        _MultiProjectForecaster = MultiProjectForecaster
    return _MultiProjectForecaster


__all__ = [
    'BaseForecaster', 'BuildingFeatures', 'ForecastResult',
    'get_lstm_forecaster', 'get_transformer_forecaster', 'get_multi_project_forecaster',
]

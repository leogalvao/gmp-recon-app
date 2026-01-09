"""
Forecasting Module - TensorFlow-based probabilistic cost forecasting.

This module provides:
- LSTM and Transformer architectures for time series forecasting
- Gaussian Mixture and Quantile outputs for uncertainty quantification
- Building parameter integration for cost prediction

Models:
- LSTMForecaster: Bidirectional LSTM with GMM output
- TransformerForecaster: Temporal Fusion Transformer with quantile outputs

Note: LSTM and Transformer models use lazy imports to avoid loading TensorFlow
at import time. Use get_* functions to access these classes.
"""

from .models.base_model import BaseForecaster, BuildingFeatures, ForecastResult
from .models import get_lstm_forecaster, get_transformer_forecaster

__all__ = [
    'BaseForecaster', 'BuildingFeatures', 'ForecastResult',
    'get_lstm_forecaster', 'get_transformer_forecaster',
]

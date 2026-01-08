"""
Feature Engineering - Data transformation and preparation.

Provides:
- Temporal feature extraction (time of year, trends)
- Static feature normalization
- Target variable scaling
"""
import numpy as np
import pickle
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for feature normalization."""
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0


class FeatureEngineer:
    """
    Feature engineering for construction cost forecasting.

    Handles:
    - Temporal cost sequence normalization
    - Static building feature scaling
    - Target variable transformation

    Attributes:
        config: Feature configuration dictionary
        temporal_scaler: Scaler for cost sequences
        static_scaler: Scaler for building features
        target_scaler: Scaler for target values
        is_fitted: Whether scalers have been fit
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.temporal_scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray
    ) -> 'FeatureEngineer':
        """
        Fit scalers to training data.

        Args:
            X_temporal: Temporal features (batch, seq_len, features)
            X_static: Static features (batch, num_features)
            y: Target values (batch,)

        Returns:
            Self for method chaining
        """
        # Reshape temporal for fitting (flatten to 2D)
        temporal_flat = X_temporal.reshape(-1, 1)
        self.temporal_scaler.fit(temporal_flat)

        # Fit static scaler
        self.static_scaler.fit(X_static)

        # Fit target scaler
        self.target_scaler.fit(y.reshape(-1, 1))

        self.is_fitted = True
        logger.info("Feature scalers fitted successfully")

        return self

    def transform(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Transform features using fitted scalers.

        Args:
            X_temporal: Temporal features
            X_static: Static features
            y: Optional target values

        Returns:
            Tuple of (transformed_temporal, transformed_static, transformed_y)
        """
        if not self.is_fitted:
            if y is not None:
                self.fit(X_temporal, X_static, y)
            else:
                raise RuntimeError("Scalers must be fit before transform")

        # Transform temporal
        orig_shape = X_temporal.shape
        temporal_flat = X_temporal.reshape(-1, 1)
        temporal_scaled = self.temporal_scaler.transform(temporal_flat)
        X_temporal_out = temporal_scaled.reshape(orig_shape)

        # Transform static
        X_static_out = self.static_scaler.transform(X_static)

        # Transform target if provided
        y_out = None
        if y is not None:
            y_out = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        return X_temporal_out, X_static_out, y_out

    def fit_transform(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(X_temporal, X_static, y)
        return self.transform(X_temporal, X_static, y)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.

        Args:
            y_scaled: Scaled predictions

        Returns:
            Predictions in original scale
        """
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()

    def extract_temporal_features(
        self,
        cost_series: np.ndarray,
        timestamps: Optional[List] = None
    ) -> np.ndarray:
        """
        Extract additional temporal features from cost series.

        Args:
            cost_series: Raw cost values (seq_len,)
            timestamps: Optional list of dates

        Returns:
            Enhanced feature array (seq_len, num_features)
        """
        features = []

        # Original values
        features.append(cost_series.reshape(-1, 1))

        # Rolling statistics
        window = 3
        if len(cost_series) >= window:
            rolling_mean = np.convolve(
                cost_series,
                np.ones(window) / window,
                mode='same'
            )
            rolling_std = np.array([
                np.std(cost_series[max(0, i-window+1):i+1])
                for i in range(len(cost_series))
            ])
            features.append(rolling_mean.reshape(-1, 1))
            features.append(rolling_std.reshape(-1, 1))

        # Lag features
        if len(cost_series) > 1:
            lag1 = np.roll(cost_series, 1)
            lag1[0] = cost_series[0]
            features.append(lag1.reshape(-1, 1))

        # Trend (simple differencing)
        if len(cost_series) > 1:
            diff = np.diff(cost_series, prepend=cost_series[0])
            features.append(diff.reshape(-1, 1))

        return np.hstack(features)

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        target_col: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from time series data.

        Args:
            data: Time series data (timesteps, features)
            sequence_length: Length of input sequences
            target_col: Column index for target variable

        Returns:
            Tuple of (X_sequences, y_targets)
        """
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length, target_col]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def save(self, path: str) -> None:
        """
        Save fitted scalers to disk.

        Args:
            path: File path for saving
        """
        state = {
            'temporal_scaler': self.temporal_scaler,
            'static_scaler': self.static_scaler,
            'target_scaler': self.target_scaler,
            'config': self.config,
            'is_fitted': self.is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Feature engineer saved to {path}")

    def load(self, path: str) -> None:
        """
        Load fitted scalers from disk.

        Args:
            path: File path for loading
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.temporal_scaler = state['temporal_scaler']
        self.static_scaler = state['static_scaler']
        self.target_scaler = state['target_scaler']
        self.config = state['config']
        self.is_fitted = state['is_fitted']
        logger.info(f"Feature engineer loaded from {path}")

    def get_feature_stats(self) -> Dict[str, Any]:
        """Get statistics about fitted features."""
        if not self.is_fitted:
            return {'fitted': False}

        return {
            'fitted': True,
            'temporal': {
                'mean': float(self.temporal_scaler.mean_[0]),
                'std': float(self.temporal_scaler.scale_[0]),
            },
            'static': {
                'means': self.static_scaler.mean_.tolist(),
                'stds': self.static_scaler.scale_.tolist(),
            },
            'target': {
                'mean': float(self.target_scaler.mean_[0]),
                'std': float(self.target_scaler.scale_[0]),
            }
        }

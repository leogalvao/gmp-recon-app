"""
LSTM-based Cost Forecaster.

Architecture:
- Bidirectional LSTM for temporal patterns
- Dense layers for building features
- Probabilistic output layer (mixture density network)

Training considers:
- Sub-jobs starting east-first
- Overlapping design phases
- ~1 month construction overlap
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from typing import Any, Dict, List, Optional
import logging

from .base_model import BaseForecaster, BuildingFeatures, ForecastResult
from ..layers.probabilistic_output import (
    GaussianMixtureLayer,
    gaussian_mixture_loss,
    sample_from_gmm,
    gmm_statistics
)

logger = logging.getLogger(__name__)


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based cost forecaster with probabilistic outputs.

    Uses a dual-input architecture:
    - Temporal branch: Bidirectional LSTM for cost history
    - Static branch: Dense layers for building parameters

    Output is a Gaussian Mixture Model for uncertainty quantification.

    Attributes:
        sequence_length: Number of historical time steps
        num_mixture_components: Number of GMM components
        lstm_units: LSTM hidden units
        dense_units: Dense layer units
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        sequence_length: int = 12,  # Months of history
        num_mixture_components: int = 3,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
    ):
        super().__init__(model_name="lstm_forecaster")
        self.sequence_length = sequence_length
        self.num_mixture_components = num_mixture_components
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model: Optional[Model] = None

    def build_model(self) -> None:
        """
        Build dual-input architecture:
        - Temporal input: Historical cost sequences
        - Static input: Building parameters
        """
        # Temporal input branch (cost history)
        temporal_input = layers.Input(
            shape=(self.sequence_length, 1),
            name='temporal_input'
        )

        # Bidirectional LSTM for temporal patterns
        x_temporal = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True),
            name='bilstm_1'
        )(temporal_input)
        x_temporal = layers.Dropout(self.dropout_rate)(x_temporal)

        x_temporal = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2),
            name='bilstm_2'
        )(x_temporal)
        x_temporal = layers.Dropout(self.dropout_rate)(x_temporal)

        # Static input branch (building features)
        static_input = layers.Input(
            shape=(7,),  # BuildingFeatures.to_array() dimension
            name='static_input'
        )

        # Process building features
        x_static = layers.Dense(
            self.dense_units,
            activation='relu',
            name='static_dense_1'
        )(static_input)
        x_static = layers.BatchNormalization()(x_static)
        x_static = layers.Dropout(self.dropout_rate)(x_static)

        x_static = layers.Dense(
            self.dense_units // 2,
            activation='relu',
            name='static_dense_2'
        )(x_static)

        # Merge branches
        merged = layers.Concatenate(name='merge')([x_temporal, x_static])

        # Final dense layers
        x = layers.Dense(
            self.dense_units,
            activation='relu',
            name='merged_dense_1'
        )(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(
            self.dense_units // 2,
            activation='relu',
            name='merged_dense_2'
        )(x)

        # Probabilistic output (Gaussian Mixture)
        output = GaussianMixtureLayer(
            num_components=self.num_mixture_components,
            name='gmm_output'
        )(x)

        self.model = Model(
            inputs=[temporal_input, static_input],
            outputs=output,
            name='lstm_cost_forecaster'
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=gaussian_mixture_loss(self.num_mixture_components)
        )

        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")

    def train(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        y_train: np.ndarray,
        X_temporal_val: Optional[np.ndarray] = None,
        X_static_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the forecaster on historical data.

        Training data should reflect:
        - East-first sub-job sequencing
        - Design phase overlaps
        - ~1 month construction overlap patterns

        Args:
            X_temporal: Temporal features (batch, seq_len, 1)
            X_static: Static features (batch, 7)
            y_train: Target values (batch,)
            X_temporal_val: Validation temporal features
            X_static_val: Validation static features
            y_val: Validation targets
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if y_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.keras',
                save_best_only=True,
                monitor='val_loss' if y_val is not None else 'loss',
                verbose=0
            )
        ]

        validation_data = None
        if all(v is not None for v in [X_temporal_val, X_static_val, y_val]):
            validation_data = ([X_temporal_val, X_static_val], y_val)

        logger.info(
            f"Training LSTM model: {len(y_train)} samples, "
            f"{epochs} max epochs, batch size {batch_size}"
        )

        history = self.model.fit(
            [X_temporal, X_static],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        logger.info(f"Training complete. Final loss: {history.history['loss'][-1]:.4f}")

        return history.history

    def predict(
        self,
        features: BuildingFeatures,
        cost_history: np.ndarray,
        confidence_level: float = 0.80
    ) -> ForecastResult:
        """
        Generate probabilistic forecast.

        Args:
            features: Building parameters
            cost_history: Historical monthly costs (shape: [sequence_length,] or [sequence_length, 1])
            confidence_level: Confidence interval width (e.g., 0.80 for 80%)

        Returns:
            ForecastResult with point estimate and uncertainty bounds
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare inputs
        cost_history = np.array(cost_history).flatten()
        if len(cost_history) < self.sequence_length:
            # Pad with zeros if insufficient history
            padding = np.zeros(self.sequence_length - len(cost_history))
            cost_history = np.concatenate([padding, cost_history])
        elif len(cost_history) > self.sequence_length:
            # Take most recent
            cost_history = cost_history[-self.sequence_length:]

        X_temporal = cost_history.reshape(1, self.sequence_length, 1)
        X_static = features.to_array().reshape(1, -1)

        # Get GMM parameters
        gmm_params = self.model.predict([X_temporal, X_static], verbose=0)[0]

        k = self.num_mixture_components
        pi = gmm_params[:k]
        mu = gmm_params[k:2*k]
        sigma = gmm_params[2*k:]

        # Compute mixture statistics
        stats = gmm_statistics(pi, mu, sigma)

        # Confidence intervals via Monte Carlo sampling
        samples = sample_from_gmm(pi, mu, sigma, n_samples=10000)
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(samples, alpha * 100)
        upper = np.percentile(samples, (1 - alpha) * 100)

        return ForecastResult(
            point_estimate=stats['mean'],
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=confidence_level,
            mean=stats['mean'],
            std=stats['std'],
            feature_importances=self._calculate_feature_importance(features)
        )

    def _calculate_feature_importance(
        self,
        features: BuildingFeatures
    ) -> Dict[str, float]:
        """
        Calculate feature importances via magnitude analysis.

        Note: This is a simplified importance estimate.
        For production, consider permutation importance or SHAP.
        """
        arr = features.to_array()
        total = np.sum(np.abs(arr)) + 1e-10

        return {
            name: float(np.abs(arr[i]) / total)
            for i, name in enumerate(self.feature_names)
        }

    def predict_multi_horizon(
        self,
        features: BuildingFeatures,
        cost_history: np.ndarray,
        horizons: int = 6,
        confidence_level: float = 0.80
    ) -> List[ForecastResult]:
        """
        Generate multi-step ahead forecasts.

        Uses autoregressive prediction: each forecast becomes
        input for the next step.

        Args:
            features: Building parameters
            cost_history: Historical costs
            horizons: Number of steps to forecast
            confidence_level: Confidence interval width

        Returns:
            List of ForecastResult for each horizon
        """
        results = []
        current_history = np.array(cost_history).flatten().copy()

        for _ in range(horizons):
            result = self.predict(features, current_history, confidence_level)
            results.append(result)

            # Append prediction to history for next step
            current_history = np.append(current_history[1:], result.point_estimate)

        return results

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        custom_objects = {
            'GaussianMixtureLayer': GaussianMixtureLayer,
            'loss_fn': gaussian_mixture_loss(self.num_mixture_components)
        }
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        base_summary = super().get_model_summary()
        base_summary.update({
            'sequence_length': self.sequence_length,
            'num_mixture_components': self.num_mixture_components,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'parameters': self.model.count_params() if self.model else 0,
        })
        return base_summary

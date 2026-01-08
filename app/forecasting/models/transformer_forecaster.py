"""
Transformer-based Cost Forecaster.

Implements Temporal Fusion Transformer (TFT) architecture
for interpretable multi-horizon forecasting.

Based on "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (Lim et al., 2019).
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Any, Dict, List, Optional
import logging

from .base_model import BaseForecaster, BuildingFeatures, ForecastResult
from ..layers.attention import MultiHeadSelfAttention, PositionalEncoding
from ..layers.gated_residual import GatedResidualNetwork

logger = logging.getLogger(__name__)


def quantile_loss(quantile: float):
    """
    Quantile regression loss (pinball loss).

    For a given quantile q:
    L(y, y_hat) = q * max(y - y_hat, 0) + (1-q) * max(y_hat - y, 0)

    Args:
        quantile: Target quantile (e.g., 0.1 for 10th percentile)

    Returns:
        Loss function compatible with Keras
    """
    def loss_fn(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(quantile * error, (quantile - 1) * error)
        )
    return loss_fn


class TransformerForecaster(BaseForecaster):
    """
    Temporal Fusion Transformer for cost forecasting.

    Features:
    - Variable selection for interpretability
    - Multi-head self-attention for temporal dependencies
    - Quantile outputs for uncertainty quantification

    Attributes:
        sequence_length: Number of historical time steps
        forecast_horizon: Number of steps to forecast
        d_model: Model dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder blocks
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        sequence_length: int = 12,
        forecast_horizon: int = 6,
        d_model: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__(model_name="transformer_forecaster")
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout_rate = dropout_rate
        self.model: Optional[Model] = None
        self.quantiles = [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles

    def build_model(self) -> None:
        """Build TFT-inspired architecture."""
        # Inputs
        temporal_input = layers.Input(
            shape=(self.sequence_length, 1),
            name='temporal_input'
        )
        static_input = layers.Input(
            shape=(7,),
            name='static_input'
        )

        # Process static features through GRN
        static_encoded = GatedResidualNetwork(
            hidden_size=self.d_model,
            dropout_rate=self.dropout_rate,
            name='static_grn'
        )(static_input)

        # Expand static context for temporal concatenation
        static_context = layers.RepeatVector(
            self.sequence_length,
            name='static_repeat'
        )(static_encoded)

        # Embed temporal input
        temporal_embedded = layers.Dense(
            self.d_model,
            name='temporal_embedding'
        )(temporal_input)

        # Add positional encoding
        temporal_embedded = PositionalEncoding(
            max_len=self.sequence_length,
            d_model=self.d_model,
            name='positional_encoding'
        )(temporal_embedded)

        # Combine temporal and static features
        combined = layers.Concatenate(name='combine')([
            temporal_embedded,
            static_context
        ])
        combined = layers.Dense(
            self.d_model,
            name='combined_projection'
        )(combined)

        # Encoder stack with attention
        encoded = combined
        for i in range(self.num_encoder_layers):
            # Self-attention
            attn_output = MultiHeadSelfAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                name=f'self_attention_{i}'
            )(encoded, encoded, encoded)
            attn_output = layers.Dropout(self.dropout_rate)(attn_output)
            encoded = layers.LayerNormalization()(encoded + attn_output)

            # Feed-forward (GRN)
            ff_output = GatedResidualNetwork(
                hidden_size=self.d_model,
                dropout_rate=self.dropout_rate,
                name=f'encoder_grn_{i}'
            )(encoded)
            encoded = layers.LayerNormalization()(encoded + ff_output)

        # Temporal aggregation
        decoded = layers.GlobalAveragePooling1D(name='temporal_pool')(encoded)

        # Final processing through GRN
        decoded = GatedResidualNetwork(
            hidden_size=self.d_model,
            output_size=self.d_model,
            dropout_rate=self.dropout_rate,
            name='decoder_grn'
        )(decoded)

        # Quantile outputs (each predicts full horizon)
        output_10 = layers.Dense(
            self.forecast_horizon,
            name='q10'
        )(decoded)
        output_50 = layers.Dense(
            self.forecast_horizon,
            name='q50'
        )(decoded)
        output_90 = layers.Dense(
            self.forecast_horizon,
            name='q90'
        )(decoded)

        self.model = Model(
            inputs=[temporal_input, static_input],
            outputs=[output_10, output_50, output_90],
            name='tft_cost_forecaster'
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'q10': quantile_loss(0.1),
                'q50': quantile_loss(0.5),
                'q90': quantile_loss(0.9),
            }
        )

        logger.info(f"Built Transformer model with {self.model.count_params()} parameters")

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
        Train the transformer model.

        Args:
            X_temporal: Temporal features (batch, seq_len, 1)
            X_static: Static features (batch, 7)
            y_train: Target values (batch,) or (batch, horizon)
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        # Ensure y has correct shape for multi-horizon
        if y_train.ndim == 1:
            y_train = np.tile(y_train.reshape(-1, 1), (1, self.forecast_horizon))

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_transformer_model.keras',
                save_best_only=True,
                verbose=0
            )
        ]

        validation_data = None
        if all(v is not None for v in [X_temporal_val, X_static_val, y_val]):
            if y_val.ndim == 1:
                y_val = np.tile(y_val.reshape(-1, 1), (1, self.forecast_horizon))
            validation_data = (
                [X_temporal_val, X_static_val],
                [y_val, y_val, y_val]  # Same target for all quantiles
            )

        logger.info(
            f"Training Transformer model: {len(y_train)} samples, "
            f"{epochs} max epochs, batch size {batch_size}"
        )

        history = self.model.fit(
            [X_temporal, X_static],
            [y_train, y_train, y_train],  # Same target for all quantiles
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        total_loss = history.history['loss'][-1]
        logger.info(f"Training complete. Final loss: {total_loss:.4f}")

        return history.history

    def predict(
        self,
        features: BuildingFeatures,
        cost_history: np.ndarray,
        confidence_level: float = 0.80
    ) -> ForecastResult:
        """
        Generate multi-horizon forecast with uncertainty.

        Args:
            features: Building parameters
            cost_history: Historical costs
            confidence_level: Confidence interval (note: model outputs 80% by default)

        Returns:
            ForecastResult with first horizon point estimate and bounds
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare inputs
        cost_history = np.array(cost_history).flatten()
        if len(cost_history) < self.sequence_length:
            padding = np.zeros(self.sequence_length - len(cost_history))
            cost_history = np.concatenate([padding, cost_history])
        elif len(cost_history) > self.sequence_length:
            cost_history = cost_history[-self.sequence_length:]

        X_temporal = cost_history.reshape(1, self.sequence_length, 1)
        X_static = features.to_array().reshape(1, -1)

        # Get quantile predictions
        q10, q50, q90 = self.model.predict([X_temporal, X_static], verbose=0)

        # Use first horizon for main result
        point_estimate = float(q50[0, 0])
        lower = float(q10[0, 0])
        upper = float(q90[0, 0])

        # Estimate std from quantile spread
        std_estimate = (upper - lower) / 2.56  # Approximate for 80% CI

        return ForecastResult(
            point_estimate=point_estimate,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            mean=point_estimate,
            std=std_estimate,
            multi_horizon=q50[0].tolist(),
            feature_importances=self._calculate_feature_importance(features)
        )

    def predict_all_horizons(
        self,
        features: BuildingFeatures,
        cost_history: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get full multi-horizon predictions with all quantiles.

        Args:
            features: Building parameters
            cost_history: Historical costs

        Returns:
            Dictionary with q10, q50, q90 arrays for all horizons
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        cost_history = np.array(cost_history).flatten()
        if len(cost_history) < self.sequence_length:
            padding = np.zeros(self.sequence_length - len(cost_history))
            cost_history = np.concatenate([padding, cost_history])
        elif len(cost_history) > self.sequence_length:
            cost_history = cost_history[-self.sequence_length:]

        X_temporal = cost_history.reshape(1, self.sequence_length, 1)
        X_static = features.to_array().reshape(1, -1)

        q10, q50, q90 = self.model.predict([X_temporal, X_static], verbose=0)

        return {
            'q10': q10[0],
            'q50': q50[0],
            'q90': q90[0],
            'horizons': list(range(1, self.forecast_horizon + 1))
        }

    def _calculate_feature_importance(
        self,
        features: BuildingFeatures
    ) -> Dict[str, float]:
        """Calculate feature importances via magnitude analysis."""
        arr = features.to_array()
        total = np.sum(np.abs(arr)) + 1e-10

        return {
            name: float(np.abs(arr[i]) / total)
            for i, name in enumerate(self.feature_names)
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'GatedResidualNetwork': GatedResidualNetwork,
            'loss_fn': quantile_loss(0.5),  # Placeholder
        }
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        base_summary = super().get_model_summary()
        base_summary.update({
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_encoder_layers': self.num_encoder_layers,
            'dropout_rate': self.dropout_rate,
            'quantiles': self.quantiles,
            'parameters': self.model.count_params() if self.model else 0,
        })
        return base_summary

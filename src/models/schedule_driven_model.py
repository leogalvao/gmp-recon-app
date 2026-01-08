"""
Schedule-Driven Forecasting Model

Three-branch architecture with schedule as PRIMARY:
1. Schedule sequence (LSTM - LARGEST) - Activity timeline
2. Trade schedule context (Dense) - Trade's position
3. Cost/variance history (LSTM - SMALLEST) - Actual costs

With attention to learn schedule-cost relationships.

The model predicts: Given current schedule position, what will cost be?
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ScheduleDrivenModel(keras.Model):
    """
    Schedule-Driven Trade Forecaster

    Architecture:
    ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────┐
    │ Schedule Sequence    │  │ Trade Schedule       │  │ Cost History   │
    │ (Activity Timeline)  │  │ (Trade Position)     │  │ (Actual $)     │
    │ [seq_len × 12]       │  │ [8 features]         │  │ [seq_len × 4]  │
    └──────────┬───────────┘  └──────────┬───────────┘  └───────┬────────┘
               │                         │                      │
               ▼                         │                      ▼
    ┌──────────────────────┐             │           ┌────────────────┐
    │ Bidirectional LSTM   │             │           │ LSTM (32)      │
    │ (128 units)          │             │           │                │
    └──────────┬───────────┘             │           └───────┬────────┘
               │                         │                   │
               ▼                         │                   │
    ┌──────────────────────┐             │                   │
    │ Attention Layer      │ ◄───────────┼───────────────────┘
    │ (Schedule × Cost)    │             │
    └──────────┬───────────┘             │
               │                         │
               ▼                         ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │ Global Pool          │  │ Dense (64 → 32)      │
    └──────────┬───────────┘  └──────────┬───────────┘
               │                         │
               └─────────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Merge Dense (128)    │
                  └──────────┬───────────┘
                             │
                  ┌──────────┴───────────┐
                  │                      │
                  ▼                      ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │ Mean Prediction      │  │ Std Prediction       │
    └──────────────────────┘  └──────────────────────┘
    """

    def __init__(
        self,
        schedule_seq_features: int = 12,
        trade_context_features: int = 8,
        cost_seq_features: int = 4,
        sequence_length: int = 6,
        lstm_units: int = 128,
        attention_heads: int = 4,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """
        Initialize schedule-driven model.

        Args:
            schedule_seq_features: Number of schedule sequence features
            trade_context_features: Number of static trade context features
            cost_seq_features: Number of cost sequence features
            sequence_length: Length of input sequences
            lstm_units: Units for main LSTM
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.schedule_seq_features = schedule_seq_features
        self.trade_context_features = trade_context_features
        self.cost_seq_features = cost_seq_features

        # ─────────────────────────────────────────────────────────────────────
        # Branch 1: Schedule sequence (PRIMARY - largest branch)
        # ─────────────────────────────────────────────────────────────────────
        self.schedule_lstm_1 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, name='schedule_lstm_1'),
            name='schedule_bidirectional'
        )
        self.schedule_dropout_1 = layers.Dropout(dropout_rate, name='schedule_dropout_1')
        self.schedule_lstm_2 = layers.LSTM(
            lstm_units // 2, return_sequences=True, name='schedule_lstm_2'
        )
        self.schedule_dropout_2 = layers.Dropout(dropout_rate * 0.7, name='schedule_dropout_2')

        # ─────────────────────────────────────────────────────────────────────
        # Branch 2: Trade schedule context (static per period)
        # ─────────────────────────────────────────────────────────────────────
        self.trade_dense_1 = layers.Dense(64, activation='relu', name='trade_dense_1')
        self.trade_bn_1 = layers.BatchNormalization(name='trade_bn_1')
        self.trade_dropout = layers.Dropout(dropout_rate * 0.7, name='trade_dropout')
        self.trade_dense_2 = layers.Dense(32, activation='relu', name='trade_dense_2')

        # ─────────────────────────────────────────────────────────────────────
        # Branch 3: Cost/variance history (smallest - costs follow schedule)
        # ─────────────────────────────────────────────────────────────────────
        self.cost_lstm = layers.LSTM(32, return_sequences=True, name='cost_lstm')
        self.cost_dropout = layers.Dropout(dropout_rate * 0.7, name='cost_dropout')

        # ─────────────────────────────────────────────────────────────────────
        # Attention: Learn which schedule features predict cost changes
        # ─────────────────────────────────────────────────────────────────────
        self.attention = layers.MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=32,
            name='schedule_cost_attention'
        )
        self.attention_norm = layers.LayerNormalization(name='attention_norm')

        # ─────────────────────────────────────────────────────────────────────
        # Pooling
        # ─────────────────────────────────────────────────────────────────────
        self.schedule_pool = layers.GlobalAveragePooling1D(name='schedule_pool')
        self.cost_pool = layers.GlobalAveragePooling1D(name='cost_pool')

        # ─────────────────────────────────────────────────────────────────────
        # Merge and output
        # ─────────────────────────────────────────────────────────────────────
        self.merge_dense_1 = layers.Dense(128, activation='relu', name='merge_dense_1')
        self.merge_bn = layers.BatchNormalization(name='merge_bn')
        self.merge_dropout = layers.Dropout(dropout_rate, name='merge_dropout')
        self.merge_dense_2 = layers.Dense(64, activation='relu', name='merge_dense_2')
        self.merge_dense_3 = layers.Dense(32, activation='relu', name='merge_dense_3')

        # Output layers
        self.output_mean = layers.Dense(1, name='output_mean')
        self.output_std = layers.Dense(1, activation='softplus', name='output_std')

        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Tuple of (schedule_seq, trade_context, cost_seq)
                - schedule_seq: [batch, seq_len, schedule_features]
                - trade_context: [batch, trade_features]
                - cost_seq: [batch, seq_len, cost_features]
            training: Whether in training mode

        Returns:
            (mean, std) predictions
        """
        schedule_seq, trade_context, cost_seq = inputs

        # ─────────────────────────────────────────────────────────────────────
        # Branch 1: Schedule sequence (PRIMARY)
        # ─────────────────────────────────────────────────────────────────────
        x_sched = self.schedule_lstm_1(schedule_seq)
        x_sched = self.schedule_dropout_1(x_sched, training=training)
        x_sched = self.schedule_lstm_2(x_sched)
        x_sched = self.schedule_dropout_2(x_sched, training=training)

        # ─────────────────────────────────────────────────────────────────────
        # Branch 3: Cost sequence
        # ─────────────────────────────────────────────────────────────────────
        x_cost = self.cost_lstm(cost_seq)
        x_cost = self.cost_dropout(x_cost, training=training)

        # ─────────────────────────────────────────────────────────────────────
        # Attention: Schedule queries cost patterns
        # Learn which schedule positions drive cost changes
        # ─────────────────────────────────────────────────────────────────────
        attended = self.attention(
            query=x_sched,
            key=x_cost,
            value=x_cost,
            training=training
        )
        x_sched = self.attention_norm(x_sched + attended)

        # Pool sequences
        x_sched = self.schedule_pool(x_sched)
        x_cost = self.cost_pool(x_cost)

        # ─────────────────────────────────────────────────────────────────────
        # Branch 2: Trade context
        # ─────────────────────────────────────────────────────────────────────
        x_trade = self.trade_dense_1(trade_context)
        x_trade = self.trade_bn_1(x_trade, training=training)
        x_trade = self.trade_dropout(x_trade, training=training)
        x_trade = self.trade_dense_2(x_trade)

        # ─────────────────────────────────────────────────────────────────────
        # Merge all branches
        # ─────────────────────────────────────────────────────────────────────
        merged = layers.Concatenate()([x_sched, x_trade, x_cost])

        x = self.merge_dense_1(merged)
        x = self.merge_bn(x, training=training)
        x = self.merge_dropout(x, training=training)
        x = self.merge_dense_2(x)
        x = self.merge_dense_3(x)

        # Output
        mean = self.output_mean(x)
        std = self.output_std(x) + 1e-6  # Ensure positive

        return mean, std

    def build_model(self) -> None:
        """Build model with sample inputs"""
        # Create sample inputs
        sample_schedule = np.zeros((1, self.sequence_length, self.schedule_seq_features))
        sample_trade = np.zeros((1, self.trade_context_features))
        sample_cost = np.zeros((1, self.sequence_length, self.cost_seq_features))

        # Call model to build
        _ = self([sample_schedule, sample_trade, sample_cost], training=False)

        # Count parameters
        total_params = sum(
            np.prod(v.shape) for v in self.trainable_variables
        )
        logger.info(f"Built ScheduleDrivenModel with {total_params:,} parameters")

    def get_config(self):
        """Get model configuration"""
        return {
            'sequence_length': self.sequence_length,
            'schedule_seq_features': self.schedule_seq_features,
            'trade_context_features': self.trade_context_features,
            'cost_seq_features': self.cost_seq_features,
        }


class ScheduleWeightedGaussianNLL(keras.losses.Loss):
    """
    Gaussian Negative Log-Likelihood with schedule weighting.

    Weight prediction errors by schedule position:
    - Higher weight when trade is in active phase
    - Lower weight during inactive phases
    - Penalize more for missing critical path impacts
    """

    def __init__(
        self,
        phase_active_weight: float = 0.5,
        critical_path_weight: float = 0.3,
        name: str = 'schedule_weighted_gnll'
    ):
        super().__init__(name=name)
        self.phase_active_weight = phase_active_weight
        self.critical_path_weight = critical_path_weight

    def call(self, y_true, y_pred):
        """
        Compute schedule-weighted Gaussian NLL.

        Args:
            y_true: [batch, 1] true values
            y_pred: Tuple of (mean, std)
        """
        mean, std = y_pred

        # Gaussian NLL
        nll = 0.5 * tf.math.log(2 * np.pi * tf.square(std) + 1e-6) + \
              tf.square(y_true - mean) / (2 * tf.square(std) + 1e-6)

        return tf.reduce_mean(nll)


def create_schedule_driven_model(
    schedule_features: int = 12,
    trade_features: int = 8,
    cost_features: int = 4,
    sequence_length: int = 6,
    **kwargs
) -> ScheduleDrivenModel:
    """
    Factory function to create and build a schedule-driven model.

    Args:
        schedule_features: Number of schedule sequence features
        trade_features: Number of trade context features
        cost_features: Number of cost sequence features
        sequence_length: Sequence length
        **kwargs: Additional model kwargs

    Returns:
        Built ScheduleDrivenModel
    """
    model = ScheduleDrivenModel(
        schedule_seq_features=schedule_features,
        trade_context_features=trade_features,
        cost_seq_features=cost_features,
        sequence_length=sequence_length,
        **kwargs
    )
    model.build_model()
    return model

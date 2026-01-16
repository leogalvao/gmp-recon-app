"""
Multi-Project Forecaster - Hierarchical transfer learning model for cross-project cost forecasting.

Implements the Phase 3 ML architecture:
- Global foundation model trained on all historical projects
- Project-specific adapter layers for fine-tuning
- Trade embeddings for capturing trade-specific cost dynamics
- Probabilistic output (Gaussian) for uncertainty quantification
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base_model import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class ProjectEmbeddingLayer(layers.Layer):
    """
    Learnable project embedding layer.

    Maps project IDs to dense vectors that capture project-specific characteristics
    like building type, region, complexity, etc.
    """

    def __init__(
        self,
        num_projects: int,
        embedding_dim: int = 32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_projects = num_projects
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(
            input_dim=num_projects + 1,  # +1 for unknown projects
            output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform',
            name='project_embedding'
        )

    def call(self, project_ids):
        """
        Args:
            project_ids: Tensor of shape (batch_size, 1) with project IDs
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        embedded = self.embedding(project_ids)
        return tf.squeeze(embedded, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_projects': self.num_projects,
            'embedding_dim': self.embedding_dim,
        })
        return config


class TradeEmbeddingLayer(layers.Layer):
    """
    Learnable trade embedding layer.

    Maps canonical trade IDs to dense vectors that capture trade-specific
    cost dynamics (e.g., concrete vs electrical vs HVAC).
    """

    def __init__(
        self,
        num_trades: int = 24,  # 23 CSI divisions + unknown
        embedding_dim: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_trades = num_trades
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(
            input_dim=num_trades + 1,  # +1 for unknown trades
            output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform',
            name='trade_embedding'
        )

    def call(self, trade_ids):
        """
        Args:
            trade_ids: Tensor of shape (batch_size, 1) with trade IDs
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        embedded = self.embedding(trade_ids)
        return tf.squeeze(embedded, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_trades': self.num_trades,
            'embedding_dim': self.embedding_dim,
        })
        return config


class TemporalEncoder(layers.Layer):
    """
    Shared temporal encoder for cost sequence patterns.

    Uses bidirectional LSTM to learn universal cost patterns across
    all construction projects. Frozen after global training.
    """

    def __init__(
        self,
        lstm_units: int = 64,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout

        self.lstm1 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
            name='bilstm_1'
        )
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False, dropout=dropout),
            name='bilstm_2'
        )
        self.layer_norm = layers.LayerNormalization(name='encoder_norm')

    def call(self, inputs, training=False):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            Encoded tensor of shape (batch_size, 2 * lstm_units)
        """
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        return self.layer_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'lstm_units': self.lstm_units,
            'dropout': self.dropout_rate,
        })
        return config


class ProjectAdapter(layers.Layer):
    """
    Project-specific adapter layer.

    Small trainable layer (~10K params) that adapts the global model
    to a specific project's cost patterns. Fine-tuned on project's
    recent data while global layers remain frozen.
    """

    def __init__(
        self,
        adapter_units: int = 32,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adapter_units = adapter_units
        self.dropout_rate = dropout

        self.dense1 = layers.Dense(adapter_units, activation='relu', name='adapter_dense1')
        self.dropout = layers.Dropout(dropout, name='adapter_dropout')
        self.dense2 = layers.Dense(adapter_units, activation='relu', name='adapter_dense2')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'adapter_units': self.adapter_units,
            'dropout': self.dropout_rate,
        })
        return config


class GaussianOutputHead(layers.Layer):
    """
    Probabilistic output head producing mean and standard deviation.

    Outputs Gaussian distribution parameters for uncertainty quantification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_layer = layers.Dense(2, name='gaussian_params')  # [mean, log_std]

    def call(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, features)
        Returns:
            Tuple of (mean, std) tensors each of shape (batch_size, 1)
        """
        output = self.output_layer(inputs)
        mean, log_std = tf.split(output, 2, axis=-1)
        std = tf.nn.softplus(log_std) + 1e-6  # Ensure positive std
        return mean, std


class MultiProjectForecaster(keras.Model, BaseForecaster):
    """
    Hierarchical model for cross-project cost forecasting.

    Architecture:
    1. Temporal Encoder: Bidirectional LSTM learning universal cost patterns
    2. Project Embedding: Captures project-specific characteristics
    3. Trade Embedding: Captures trade-specific cost dynamics
    4. Feature Fusion: Combines temporal, project, and trade features
    5. Project Adapter: Fine-tunable layer for project-specific adaptation
    6. Gaussian Output: Mean + std for probabilistic forecasting

    Training Strategy:
    - Global training: Train all layers on historical data from all projects
    - Fine-tuning: Freeze encoder/embeddings, train only adapter on project data
    """

    def __init__(
        self,
        num_projects: int,
        num_trades: int = 24,
        seq_len: int = 12,
        feature_dim: int = 5,  # cost_per_sf, cumulative, budget, pct_complete, schedule_elapsed
        project_embed_dim: int = 32,
        trade_embed_dim: int = 16,
        lstm_units: int = 64,
        adapter_units: int = 32,
        dropout: float = 0.2,
        model_name: str = "multi_project_forecaster",
    ):
        keras.Model.__init__(self)
        BaseForecaster.__init__(self, model_name)

        self.num_projects = num_projects
        self.num_trades = num_trades
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.project_embed_dim = project_embed_dim
        self.trade_embed_dim = trade_embed_dim
        self.lstm_units = lstm_units
        self.adapter_units = adapter_units
        self.dropout_rate = dropout

        # Embeddings
        self.project_embedding = ProjectEmbeddingLayer(
            num_projects, project_embed_dim, name='project_embed'
        )
        self.trade_embedding = TradeEmbeddingLayer(
            num_trades, trade_embed_dim, name='trade_embed'
        )

        # Shared temporal encoder
        self.temporal_encoder = TemporalEncoder(
            lstm_units, dropout, name='temporal_encoder'
        )

        # Feature fusion
        self.fusion = layers.Dense(64, activation='relu', name='fusion')

        # Project-specific adapter
        self.adapter = ProjectAdapter(adapter_units, dropout, name='adapter')

        # Probabilistic output
        self.output_head = GaussianOutputHead(name='output_head')

        # Track training state
        self.is_globally_trained = False
        self.finetuned_projects = set()

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: Tuple of (seq_features, project_ids, trade_ids)
                - seq_features: (batch, seq_len, feature_dim)
                - project_ids: (batch, 1)
                - trade_ids: (batch, 1)
            training: Whether in training mode

        Returns:
            Tuple of (mean, std) for Gaussian distribution
        """
        seq_features, project_ids, trade_ids = inputs

        # Encode temporal sequence
        temporal_encoding = self.temporal_encoder(seq_features, training=training)

        # Get embeddings
        proj_embed = self.project_embedding(project_ids)
        trade_embed = self.trade_embedding(trade_ids)

        # Fuse all features
        combined = tf.concat([temporal_encoding, proj_embed, trade_embed], axis=-1)
        fused = self.fusion(combined)

        # Project-specific adaptation
        adapted = self.adapter(fused, training=training)

        # Probabilistic output
        mean, std = self.output_head(adapted)

        return mean, std

    def build_model(self):
        """Build model by running a forward pass with dummy data."""
        dummy_seq = tf.zeros((1, self.seq_len, self.feature_dim))
        dummy_proj = tf.zeros((1, 1), dtype=tf.int32)
        dummy_trade = tf.zeros((1, 1), dtype=tf.int32)
        _ = self((dummy_seq, dummy_proj, dummy_trade), training=False)
        logger.info(f"Model built with {self.count_params()} parameters")

    def train(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        y_train: np.ndarray,
        X_temporal_val: Optional[np.ndarray] = None,
        X_static_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model (wrapper for fit method).

        Note: For multi-project training, use train_global() instead.
        This method is for single-project backward compatibility.
        """
        raise NotImplementedError(
            "Use train_global() for multi-project training or "
            "finetune_for_project() for project-specific fine-tuning"
        )

    def train_global(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the global foundation model on all historical projects.

        Args:
            train_dataset: TF Dataset yielding ((seq_features, proj_ids, trade_ids), targets)
            val_dataset: Validation dataset
            epochs: Maximum training epochs
            learning_rate: Learning rate for Adam optimizer
            early_stopping_patience: Epochs to wait for improvement
            checkpoint_path: Path to save best model

        Returns:
            Training history
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
        ]
        if checkpoint_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
            )

        # Compile with Gaussian NLL loss
        self.compile(
            optimizer=optimizer,
            loss=self._gaussian_nll_loss,
        )

        # Train
        history = self.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )

        self.is_globally_trained = True
        self.is_trained = True

        logger.info(f"Global training complete after {len(history.history['loss'])} epochs")

        return history.history

    def finetune_for_project(
        self,
        project_id: int,
        project_dataset: tf.data.Dataset,
        epochs: int = 20,
        learning_rate: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune adapter layers for a specific project.

        Freezes global components (encoder, embeddings) and only
        trains adapter and output head on project-specific data.

        Args:
            project_id: ID of project to fine-tune for
            project_dataset: TF Dataset for the project
            epochs: Fine-tuning epochs
            learning_rate: Learning rate (lower than global training)

        Returns:
            Fine-tuning history
        """
        if not self.is_globally_trained:
            logger.warning("Fine-tuning without global training - results may be poor")

        # Freeze global components
        self.temporal_encoder.trainable = False
        self.project_embedding.trainable = False
        self.trade_embedding.trainable = False
        self.fusion.trainable = False

        # Keep adapter and output head trainable
        self.adapter.trainable = True
        self.output_head.trainable = True

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.compile(
            optimizer=optimizer,
            loss=self._gaussian_nll_loss,
        )

        history = self.fit(
            project_dataset,
            epochs=epochs,
        )

        self.finetuned_projects.add(project_id)

        # Restore trainability for future global training
        self._unfreeze_all()

        logger.info(f"Fine-tuning complete for project {project_id}")

        return history.history

    def _unfreeze_all(self):
        """Restore all layers to trainable state."""
        self.temporal_encoder.trainable = True
        self.project_embedding.trainable = True
        self.trade_embedding.trainable = True
        self.fusion.trainable = True
        self.adapter.trainable = True
        self.output_head.trainable = True

    @staticmethod
    def _gaussian_nll_loss(y_true, y_pred):
        """
        Gaussian Negative Log-Likelihood loss.

        Args:
            y_true: Ground truth values (batch, 1)
            y_pred: Tuple of (mean, std) tensors
        """
        # y_pred comes from call() as tuple, but Keras passes concatenated tensor
        # Split it back
        mean, log_std = tf.split(y_pred, 2, axis=-1)
        std = tf.nn.softplus(log_std) + 1e-6

        variance = tf.square(std)
        log_likelihood = -0.5 * (
            tf.math.log(2 * np.pi * variance) +
            tf.square(y_true - mean) / variance
        )
        return -tf.reduce_mean(log_likelihood)

    def predict_with_uncertainty(
        self,
        seq_features: np.ndarray,
        project_id: int,
        trade_id: int,
        confidence_level: float = 0.80,
    ) -> ForecastResult:
        """
        Generate forecast with uncertainty quantification.

        Args:
            seq_features: Sequence features (seq_len, feature_dim)
            project_id: Project ID
            trade_id: Canonical trade ID
            confidence_level: Confidence interval width

        Returns:
            ForecastResult with point estimate and bounds
        """
        # Prepare inputs
        seq_batch = np.expand_dims(seq_features, 0)
        proj_batch = np.array([[project_id]], dtype=np.int32)
        trade_batch = np.array([[trade_id]], dtype=np.int32)

        # Get prediction
        mean, std = self((seq_batch, proj_batch, trade_batch), training=False)

        mean_val = float(mean.numpy()[0, 0])
        std_val = float(std.numpy()[0, 0])

        # Compute confidence bounds
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = mean_val - z_score * std_val
        upper_bound = mean_val + z_score * std_val

        return ForecastResult(
            point_estimate=mean_val,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            mean=mean_val,
            std=std_val,
        )

    def predict(
        self,
        features,  # BuildingFeatures - not used for multi-project
        cost_history: np.ndarray,
        confidence_level: float = 0.80,
    ) -> ForecastResult:
        """
        Predict (backward compatibility interface).

        For multi-project forecasting, use predict_with_uncertainty() instead.
        """
        raise NotImplementedError(
            "Use predict_with_uncertainty() with project_id and trade_id"
        )

    def save(self, path: str) -> None:
        """Save model weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        self.save_weights(str(path / "model.weights.h5"))

        # Save config
        config = {
            'num_projects': self.num_projects,
            'num_trades': self.num_trades,
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim,
            'project_embed_dim': self.project_embed_dim,
            'trade_embed_dim': self.trade_embed_dim,
            'lstm_units': self.lstm_units,
            'adapter_units': self.adapter_units,
            'dropout': self.dropout_rate,
            'is_globally_trained': self.is_globally_trained,
            'finetuned_projects': list(self.finetuned_projects),
        }
        import json
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights and config."""
        path = Path(path)

        # Load config
        import json
        with open(path / "config.json", 'r') as f:
            config = json.load(f)

        self.is_globally_trained = config.get('is_globally_trained', False)
        self.finetuned_projects = set(config.get('finetuned_projects', []))

        # Build model first
        self.build_model()

        # Load weights
        self.load_weights(str(path / "model.weights.h5"))
        self.is_trained = True

        logger.info(f"Model loaded from {path}")

    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'num_projects': self.num_projects,
            'num_trades': self.num_trades,
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim,
            'project_embed_dim': self.project_embed_dim,
            'trade_embed_dim': self.trade_embed_dim,
            'lstm_units': self.lstm_units,
            'adapter_units': self.adapter_units,
            'dropout': self.dropout_rate,
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get detailed model summary."""
        base_summary = BaseForecaster.get_model_summary(self)
        base_summary.update({
            'num_projects': self.num_projects,
            'num_trades': self.num_trades,
            'seq_len': self.seq_len,
            'is_globally_trained': self.is_globally_trained,
            'finetuned_projects_count': len(self.finetuned_projects),
            'total_params': self.count_params() if self.built else 0,
        })
        return base_summary

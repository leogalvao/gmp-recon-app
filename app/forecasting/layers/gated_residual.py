"""
Gated Residual Network - Core building block for Temporal Fusion Transformer.

Implements the GRN from "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (Lim et al., 2019).
"""
import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional


class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN) for variable selection.

    Core building block of Temporal Fusion Transformer providing:
    - Non-linear processing with ELU activation
    - Gated Linear Unit (GLU) for adaptive information flow
    - Skip connection with optional projection

    Attributes:
        hidden_size: Size of hidden layers
        output_size: Size of output (defaults to hidden_size)
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int = 64,
        output_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """Build layer weights."""
        # Main pathway
        self.dense1 = layers.Dense(
            self.hidden_size,
            activation='elu',
            name='dense1'
        )
        self.dense2 = layers.Dense(
            self.hidden_size,
            name='dense2'
        )

        # GLU gate
        self.glu_dense = layers.Dense(
            self.hidden_size * 2,
            name='glu'
        )

        # Output projection
        self.output_dense = layers.Dense(
            self.output_size,
            name='output'
        )

        # Skip connection projection (if dimensions differ)
        self.skip_dense = None
        input_dim = input_shape[-1]
        if isinstance(input_dim, int) and input_dim != self.output_size:
            self.skip_dense = layers.Dense(
                self.output_size,
                name='skip_projection'
            )

        # Regularization
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, x, context=None, training=None):
        """
        Forward pass.

        Args:
            x: Input tensor
            context: Optional context tensor (concatenated with x)
            training: Whether in training mode

        Returns:
            Processed tensor with skip connection
        """
        # Store original for skip connection
        original = x

        # Concatenate context if provided
        if context is not None:
            x = tf.concat([x, context], axis=-1)

        # Main pathway
        hidden = self.dense1(x)
        hidden = self.dropout(hidden, training=training)
        hidden = self.dense2(hidden)

        # Gated Linear Unit
        glu_output = self.glu_dense(hidden)
        glu_a, glu_b = tf.split(glu_output, 2, axis=-1)
        gated = glu_a * tf.sigmoid(glu_b)

        # Output projection
        output = self.output_dense(gated)

        # Skip connection
        if self.skip_dense is not None:
            original = self.skip_dense(original)

        output = output + original

        # Layer normalization
        return self.layer_norm(output)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
        })
        return config


class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network for selecting relevant features.

    Uses GRN with softmax to weight feature importance.

    Attributes:
        num_features: Number of input features
        hidden_size: GRN hidden size
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """Build layer weights."""
        # GRN for each feature
        self.feature_grns = [
            GatedResidualNetwork(
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                name=f'feature_grn_{i}'
            )
            for i in range(self.num_features)
        ]

        # Variable selection weights
        self.weight_grn = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            output_size=self.num_features,
            dropout_rate=self.dropout_rate,
            name='weight_grn'
        )

        self.softmax = layers.Softmax(axis=-1)
        super().build(input_shape)

    def call(self, x, training=None):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, num_features, feature_dim)
            training: Whether in training mode

        Returns:
            Tuple of (weighted_sum, variable_weights)
        """
        # Flatten for weight computation
        batch_size = tf.shape(x)[0]
        flattened = tf.reshape(x, (batch_size, -1))

        # Compute variable selection weights
        weight_input = self.weight_grn(flattened, training=training)
        variable_weights = self.softmax(weight_input)

        # Process each feature through its GRN
        processed_features = []
        for i in range(self.num_features):
            feature = x[:, i, :]
            processed = self.feature_grns[i](feature, training=training)
            processed_features.append(processed)

        # Stack processed features
        stacked = tf.stack(processed_features, axis=1)

        # Apply variable weights
        weights_expanded = tf.expand_dims(variable_weights, -1)
        weighted_sum = tf.reduce_sum(stacked * weights_expanded, axis=1)

        return weighted_sum, variable_weights

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
        })
        return config

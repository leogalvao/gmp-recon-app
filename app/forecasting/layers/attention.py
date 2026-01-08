"""
Attention Layers - Multi-head self-attention and positional encoding.

Implements Transformer-style attention mechanisms for temporal forecasting.
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for sequences.

    Adds position information to input embeddings using sine and cosine
    functions at different frequencies, as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        max_len: Maximum sequence length
        d_model: Model dimension (embedding size)
    """

    def __init__(self, max_len: int = 100, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = None

    def build(self, input_shape):
        """Precompute positional encodings."""
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]

        # Compute angles
        angles = positions / np.power(10000, (2 * (dims // 2)) / self.d_model)

        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        # Add batch dimension and convert to tensor
        self.pos_encoding = tf.constant(
            angles[np.newaxis, ...],
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads
    for learning different attention patterns.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        depth: Dimension per head (d_model // num_heads)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

    def build(self, input_shape):
        """Build query, key, value projections."""
        self.wq = layers.Dense(self.d_model, name='query')
        self.wk = layers.Dense(self.d_model, name='key')
        self.wv = layers.Dense(self.d_model, name='value')
        self.dense = layers.Dense(self.d_model, name='output')
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).

        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            batch_size: Batch size

        Returns:
            Tensor of shape (batch, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

        Args:
            q: Query tensor (batch, heads, seq_q, depth)
            k: Key tensor (batch, heads, seq_k, depth)
            v: Value tensor (batch, heads, seq_v, depth)
            mask: Optional attention mask

        Returns:
            Output tensor and attention weights
        """
        # Compute attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale by sqrt(depth)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax for attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention to values
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q, mask=None, return_attention=False):
        """
        Forward pass.

        Args:
            v: Value tensor
            k: Key tensor
            q: Query tensor
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor, optionally with attention weights
        """
        batch_size = tf.shape(q)[0]

        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split into heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        # Reshape back
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model)
        )

        # Final projection
        output = self.dense(concat_attention)

        if return_attention:
            return output, attention_weights
        return output

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


class CausalSelfAttention(MultiHeadSelfAttention):
    """
    Causal (masked) self-attention for autoregressive models.

    Prevents the model from attending to future positions.
    """

    def call(self, x, return_attention=False):
        """
        Forward pass with causal mask.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor, optionally with attention weights
        """
        seq_len = tf.shape(x)[1]

        # Create causal mask (lower triangular)
        mask = 1 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )

        return super().call(x, x, x, mask=mask, return_attention=return_attention)

"""
Probabilistic Output Layers - Mixture Density Network components.

Implements Gaussian Mixture Model output for uncertainty quantification.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class GaussianMixtureLayer(layers.Layer):
    """
    Mixture Density Network (MDN) output layer.

    Outputs parameters for a Gaussian mixture model:
    - Mixture weights (pi): Softmax normalized component weights
    - Means (mu): Component mean values
    - Standard deviations (sigma): Component standard deviations

    The output is concatenated as: [pi_1...pi_k, mu_1...mu_k, sigma_1...sigma_k]

    Attributes:
        num_components: Number of mixture components (k)
    """

    def __init__(self, num_components: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components

    def build(self, input_shape):
        """Build the layer's weights."""
        self.dense_pi = layers.Dense(
            self.num_components,
            activation='softmax',
            name='mixture_weights'
        )
        self.dense_mu = layers.Dense(
            self.num_components,
            name='mixture_means'
        )
        self.dense_sigma = layers.Dense(
            self.num_components,
            activation='softplus',
            name='mixture_stds'
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, features)

        Returns:
            Tensor of shape (batch, 3 * num_components)
            containing [pi, mu, sigma]
        """
        pi = self.dense_pi(inputs)
        mu = self.dense_mu(inputs)
        sigma = self.dense_sigma(inputs) + 1e-6  # Numerical stability

        return tf.concat([pi, mu, sigma], axis=-1)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_components': self.num_components
        })
        return config


def gaussian_mixture_loss(num_components: int = 3):
    """
    Negative log-likelihood loss for Gaussian mixture model.

    Computes the NLL of observing y_true under the predicted GMM:
    NLL = -log(sum_k pi_k * N(y | mu_k, sigma_k))

    Args:
        num_components: Number of mixture components

    Returns:
        Loss function compatible with Keras
    """
    def loss_fn(y_true, y_pred):
        # Parse predictions
        k = num_components
        pi = y_pred[:, :k]
        mu = y_pred[:, k:2*k]
        sigma = y_pred[:, 2*k:]

        # Expand y_true for broadcasting
        y_true_expanded = tf.expand_dims(y_true, -1)  # (batch, 1)

        # Compute Gaussian PDF for each component
        # N(y | mu, sigma) = exp(-0.5 * ((y - mu) / sigma)^2) / (sigma * sqrt(2*pi))
        exponent = -0.5 * tf.square((y_true_expanded - mu) / sigma)
        gaussian = tf.exp(exponent) / (sigma * tf.sqrt(2 * np.pi))

        # Weighted sum of Gaussians
        weighted_sum = tf.reduce_sum(pi * gaussian, axis=-1)

        # Negative log likelihood
        nll = -tf.math.log(weighted_sum + 1e-10)

        return tf.reduce_mean(nll)

    return loss_fn


def sample_from_gmm(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_samples: int = 10000
) -> np.ndarray:
    """
    Sample from a Gaussian Mixture Model.

    Args:
        pi: Component weights (k,)
        mu: Component means (k,)
        sigma: Component standard deviations (k,)
        n_samples: Number of samples to draw

    Returns:
        Array of samples (n_samples,)
    """
    k = len(pi)
    samples = []

    for _ in range(n_samples):
        # Sample component according to weights
        component = np.random.choice(k, p=pi)
        # Sample from selected component
        sample = np.random.normal(mu[component], sigma[component])
        samples.append(sample)

    return np.array(samples)


def gmm_statistics(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray
) -> dict:
    """
    Compute statistics of a Gaussian Mixture Model.

    Args:
        pi: Component weights (k,)
        mu: Component means (k,)
        sigma: Component standard deviations (k,)

    Returns:
        Dictionary with mean, variance, std, and mode
    """
    # Mixture mean: E[X] = sum_k pi_k * mu_k
    mixture_mean = np.sum(pi * mu)

    # Mixture variance: Var[X] = sum_k pi_k * (sigma_k^2 + mu_k^2) - E[X]^2
    mixture_var = np.sum(pi * (sigma**2 + mu**2)) - mixture_mean**2
    mixture_std = np.sqrt(mixture_var)

    # Mode: approximate as mean of highest-weight component
    mode_component = np.argmax(pi)
    mode = mu[mode_component]

    return {
        'mean': float(mixture_mean),
        'variance': float(mixture_var),
        'std': float(mixture_std),
        'mode': float(mode),
        'mode_weight': float(pi[mode_component]),
    }

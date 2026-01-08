"""
Base Forecasting Model - Abstract interface for all forecasters.

Implements:
- Probabilistic output (mean + uncertainty intervals)
- Building parameter feature integration
- Historical pattern learning
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ForecastResult:
    """
    Container for forecast outputs with uncertainty quantification.

    Attributes:
        point_estimate: Best estimate (mean or median)
        lower_bound: Lower confidence bound (e.g., 10th percentile)
        upper_bound: Upper confidence bound (e.g., 90th percentile)
        confidence_level: Confidence interval width (e.g., 0.80 for 80%)
        mean: Distribution mean
        std: Distribution standard deviation
        feature_importances: Optional feature importance scores
        multi_horizon: Optional multi-step forecasts
    """

    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.80

    # Detailed distribution parameters
    mean: float = 0.0
    std: float = 0.0

    # Feature importances
    feature_importances: Optional[Dict[str, float]] = None

    # Multi-horizon forecasts (for multi-step models)
    multi_horizon: Optional[List[float]] = None

    def uncertainty_range(self) -> float:
        """Calculate the uncertainty range (width of confidence interval)."""
        return self.upper_bound - self.lower_bound

    def coefficient_of_variation(self) -> float:
        """Calculate CV (std/mean) as a normalized uncertainty measure."""
        if abs(self.mean) < 1e-10:
            return 0.0
        return self.std / abs(self.mean)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'point_estimate': self.point_estimate,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_level': self.confidence_level,
            'mean': self.mean,
            'std': self.std,
            'uncertainty_range': self.uncertainty_range(),
            'coefficient_of_variation': self.coefficient_of_variation(),
            'feature_importances': self.feature_importances,
            'multi_horizon': self.multi_horizon,
        }


@dataclass
class BuildingFeatures:
    """
    Standardized building features for forecasting.

    Used as static features alongside temporal cost history.

    Attributes:
        sqft: Building square footage
        stories: Number of floors
        has_green_roof: Green roof presence (complexity indicator)
        rooftop_units_qty: Number of rooftop HVAC units
        fall_anchor_count: Number of fall protection anchors
        sqft_per_story: Derived - average sqft per floor
        complexity_score: Derived - overall complexity (0-1)
    """

    sqft: float
    stories: int
    has_green_roof: bool
    rooftop_units_qty: int
    fall_anchor_count: int

    # Derived features (computed in __post_init__)
    sqft_per_story: float = field(init=False)
    complexity_score: float = field(init=False)

    def __post_init__(self):
        """Compute derived features."""
        # Square footage per story
        if self.stories > 0:
            self.sqft_per_story = self.sqft / self.stories
        else:
            self.sqft_per_story = 0.0

        # Calculate complexity score (0-1)
        self.complexity_score = (
            (1.0 if self.has_green_roof else 0.0) * 0.3 +
            min(self.rooftop_units_qty / 10.0, 1.0) * 0.4 +
            min(self.fall_anchor_count / 50.0, 1.0) * 0.3
        )

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array for model input.

        Returns:
            Array with shape (7,) containing all features
        """
        return np.array([
            self.sqft,
            float(self.stories),
            float(self.has_green_roof),
            float(self.rooftop_units_qty),
            float(self.fall_anchor_count),
            self.sqft_per_story,
            self.complexity_score,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BuildingFeatures':
        """Create BuildingFeatures from dictionary."""
        return cls(
            sqft=float(data.get('sqft', 0)),
            stories=int(data.get('stories', 1)),
            has_green_roof=bool(data.get('has_green_roof', False)),
            rooftop_units_qty=int(data.get('rooftop_units_qty', 0)),
            fall_anchor_count=int(data.get('fall_anchor_count', 0)),
        )


class BaseForecaster(ABC):
    """
    Abstract base class for cost forecasters.

    All forecasting models should inherit from this class and
    implement the abstract methods.

    Attributes:
        model_name: Identifier for the model
        is_trained: Whether the model has been trained
        feature_names: List of feature names for interpretation
    """

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.is_trained = False
        self.feature_names = [
            'sqft', 'stories', 'has_green_roof',
            'rooftop_units_qty', 'fall_anchor_count',
            'sqft_per_story', 'complexity_score'
        ]

    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
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
        Train the model.

        Args:
            X_temporal: Temporal features (batch, seq_len, features)
            X_static: Static building features (batch, features)
            y_train: Target values
            X_temporal_val: Validation temporal features
            X_static_val: Validation static features
            y_val: Validation targets
            **kwargs: Additional training arguments

        Returns:
            Training history dictionary
        """
        pass

    @abstractmethod
    def predict(
        self,
        features: BuildingFeatures,
        cost_history: np.ndarray,
        confidence_level: float = 0.80
    ) -> ForecastResult:
        """
        Generate forecast with uncertainty.

        Args:
            features: Building parameters
            cost_history: Historical monthly costs (seq_len, 1)
            confidence_level: Confidence interval width

        Returns:
            ForecastResult with point estimate and bounds
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
        }

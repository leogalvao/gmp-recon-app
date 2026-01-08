"""
Training Pipeline - End-to-end model training orchestration.

Features:
- Configurable training from YAML
- Feature engineering integration
- Cross-validation with time series splits
- Model selection (LSTM vs Transformer)
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

from ...forecasting.models.lstm_forecaster import LSTMForecaster
from ...forecasting.models.transformer_forecaster import TransformerForecaster
from ...forecasting.models.base_model import BuildingFeatures, BaseForecaster
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Parsed training configuration.

    Loaded from YAML config file.
    """
    # Data parameters
    lookback_months: int = 12
    forecast_horizon: int = 6
    train_split: float = 0.8
    val_split: float = 0.15

    # Model config
    architecture: str = "lstm"  # "lstm" or "transformer"
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Training params
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 10

    # Feature config
    feature_config: Dict[str, Any] = field(default_factory=dict)

    # Sub-job constraints
    construction_overlap_target: int = 30

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)

        return cls(
            lookback_months=raw_config.get('data', {}).get('lookback_months', 12),
            forecast_horizon=raw_config.get('data', {}).get('forecast_horizon_months', 6),
            train_split=raw_config.get('data', {}).get('train_test_split', 0.8),
            val_split=raw_config.get('data', {}).get('validation_split', 0.15),
            architecture=raw_config.get('model', {}).get('architecture', 'lstm'),
            model_params=raw_config.get('model', {}).get(
                raw_config.get('model', {}).get('architecture', 'lstm'), {}
            ),
            epochs=raw_config.get('training', {}).get('epochs', 100),
            batch_size=raw_config.get('training', {}).get('batch_size', 32),
            learning_rate=raw_config.get('training', {}).get('learning_rate', 0.001),
            patience=raw_config.get('training', {}).get('early_stopping', {}).get('patience', 10),
            feature_config=raw_config.get('features', {}),
            construction_overlap_target=raw_config.get('data', {}).get(
                'sub_job_sequencing', {}
            ).get('construction_overlap_days', {}).get('target', 30),
        )


class TrainingPipeline:
    """
    Orchestrates model training with configuration management.

    Provides:
    - Data preparation from historical records
    - Model initialization based on architecture choice
    - Training with validation
    - Model and feature scaler persistence
    """

    def __init__(
        self,
        config: Optional[Union[str, TrainingConfig]] = None
    ):
        """
        Initialize training pipeline.

        Args:
            config: Path to YAML config file or TrainingConfig object
        """
        if config is None:
            self.config = TrainingConfig()
        elif isinstance(config, str):
            self.config = TrainingConfig.from_yaml(config)
        else:
            self.config = config

        self.feature_engineer = FeatureEngineer(self.config.feature_config)
        self.model: Optional[BaseForecaster] = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model based on architecture configuration."""
        if self.config.architecture == "lstm":
            self.model = LSTMForecaster(
                sequence_length=self.config.lookback_months,
                lstm_units=self.config.model_params.get('units', 64),
                dropout_rate=self.config.model_params.get('dropout', 0.2),
            )
        elif self.config.architecture == "transformer":
            self.model = TransformerForecaster(
                sequence_length=self.config.lookback_months,
                forecast_horizon=self.config.forecast_horizon,
                d_model=self.config.model_params.get('d_model', 64),
                num_heads=self.config.model_params.get('num_heads', 4),
            )
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

        self.model.build_model()
        logger.info(f"Initialized {self.config.architecture} model")

    def prepare_data(
        self,
        historical_costs: pd.DataFrame,
        building_data: pd.DataFrame,
        cost_col: str = 'cost',
        date_col: str = 'date',
        project_col: str = 'project_id'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from historical records.

        Args:
            historical_costs: DataFrame with columns [project_id, date, cost]
            building_data: DataFrame with building parameters
            cost_col: Name of cost column
            date_col: Name of date column
            project_col: Name of project ID column

        Returns:
            X_temporal_train, X_static_train, y_train,
            X_temporal_val, X_static_val, y_val
        """
        sequences = []
        targets = []
        static_features = []

        for project_id in historical_costs[project_col].unique():
            project_costs = historical_costs[
                historical_costs[project_col] == project_id
            ].sort_values(date_col)

            if len(project_costs) < self.config.lookback_months + 1:
                logger.debug(f"Skipping project {project_id}: insufficient data")
                continue

            # Get building features
            building_rows = building_data[
                building_data[project_col] == project_id
            ]
            if len(building_rows) == 0:
                logger.debug(f"Skipping project {project_id}: no building data")
                continue

            building_row = building_rows.iloc[0]

            features = BuildingFeatures(
                sqft=float(building_row.get('sqft', building_row.get('square_feet', 0))),
                stories=int(building_row.get('stories', 1)),
                has_green_roof=bool(building_row.get('has_green_roof', False)),
                rooftop_units_qty=int(building_row.get('rooftop_units_qty', 0)),
                fall_anchor_count=int(building_row.get('fall_anchor_count', 0)),
            )

            # Create sliding window sequences
            costs = project_costs[cost_col].values
            for i in range(len(costs) - self.config.lookback_months):
                seq = costs[i:i + self.config.lookback_months]
                target = costs[i + self.config.lookback_months]

                sequences.append(seq)
                targets.append(target)
                static_features.append(features.to_array())

        if len(sequences) == 0:
            raise ValueError("No valid sequences could be created from the data")

        # Convert to numpy arrays
        X_temporal = np.array(sequences)[:, :, np.newaxis]
        X_static = np.array(static_features)
        y = np.array(targets)

        logger.info(f"Created {len(y)} training samples from {historical_costs[project_col].nunique()} projects")

        # Normalize features
        X_temporal, X_static, y = self.feature_engineer.fit_transform(
            X_temporal, X_static, y
        )

        # Split data (time-aware: don't shuffle)
        n_samples = len(y)
        train_idx = int(n_samples * self.config.train_split)
        val_split_from_train = int(train_idx * (1 - self.config.val_split))

        X_temporal_train = X_temporal[:val_split_from_train]
        X_static_train = X_static[:val_split_from_train]
        y_train = y[:val_split_from_train]

        X_temporal_val = X_temporal[val_split_from_train:train_idx]
        X_static_val = X_static[val_split_from_train:train_idx]
        y_val = y[val_split_from_train:train_idx]

        logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")

        return (
            X_temporal_train, X_static_train, y_train,
            X_temporal_val, X_static_val, y_val
        )

    def train(
        self,
        historical_costs: pd.DataFrame,
        building_data: pd.DataFrame,
        cost_col: str = 'cost',
        date_col: str = 'date',
        project_col: str = 'project_id'
    ) -> Dict[str, List[float]]:
        """
        Execute full training pipeline.

        Args:
            historical_costs: DataFrame with cost history
            building_data: DataFrame with building parameters
            cost_col: Name of cost column
            date_col: Name of date column
            project_col: Name of project ID column

        Returns:
            Training history dictionary
        """
        logger.info("Starting training pipeline...")

        # Prepare data
        (X_temporal_train, X_static_train, y_train,
         X_temporal_val, X_static_val, y_val) = self.prepare_data(
            historical_costs, building_data,
            cost_col, date_col, project_col
        )

        # Train model
        history = self.model.train(
            X_temporal=X_temporal_train,
            X_static=X_static_train,
            y_train=y_train,
            X_temporal_val=X_temporal_val,
            X_static_val=X_static_val,
            y_val=y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            patience=self.config.patience,
        )

        logger.info("Training complete!")
        return history

    def predict(
        self,
        features: BuildingFeatures,
        cost_history: np.ndarray,
        confidence_level: float = 0.80
    ):
        """
        Generate forecast using trained model.

        Args:
            features: Building parameters
            cost_history: Historical cost values
            confidence_level: Confidence interval width

        Returns:
            ForecastResult
        """
        if not self.model.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Scale the cost history
        cost_history = np.array(cost_history).flatten()
        cost_scaled, _, _ = self.feature_engineer.transform(
            cost_history.reshape(1, -1, 1),
            features.to_array().reshape(1, -1),
            None
        )

        result = self.model.predict(
            features=features,
            cost_history=cost_scaled.flatten(),
            confidence_level=confidence_level
        )

        # Inverse transform the predictions
        result.point_estimate = float(
            self.feature_engineer.inverse_transform_target(
                np.array([result.point_estimate])
            )[0]
        )
        result.lower_bound = float(
            self.feature_engineer.inverse_transform_target(
                np.array([result.lower_bound])
            )[0]
        )
        result.upper_bound = float(
            self.feature_engineer.inverse_transform_target(
                np.array([result.upper_bound])
            )[0]
        )
        result.mean = result.point_estimate

        return result

    def save(self, model_path: str) -> None:
        """
        Save trained model and feature engineer.

        Args:
            model_path: Path for model file (features saved alongside)
        """
        self.model.save(model_path)

        # Save feature engineering state
        feature_path = model_path.replace('.keras', '_features.pkl')
        if feature_path == model_path:
            feature_path = model_path + '_features.pkl'
        self.feature_engineer.save(feature_path)

        logger.info(f"Pipeline saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load trained model and feature engineer.

        Args:
            model_path: Path to model file
        """
        self.model.load(model_path)

        # Load feature engineering state
        feature_path = model_path.replace('.keras', '_features.pkl')
        if feature_path == model_path:
            feature_path = model_path + '_features.pkl'
        self.feature_engineer.load(feature_path)

        logger.info(f"Pipeline loaded from {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'config': {
                'architecture': self.config.architecture,
                'lookback_months': self.config.lookback_months,
                'forecast_horizon': self.config.forecast_horizon,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
            },
            'model': self.model.get_model_summary() if self.model else None,
            'features': self.feature_engineer.get_feature_stats(),
        }

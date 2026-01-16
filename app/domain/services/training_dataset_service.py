"""
Training Dataset Service - Builds TensorFlow datasets from canonical cost features.

Handles:
1. Temporal sequence generation with proper lookback windows
2. Time-based train/validation splitting (leakage prevention)
3. Project-stratified sampling
4. Feature normalization
"""
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.models import (
    Project,
    CanonicalCostFeature,
    CanonicalTrade,
    TrainingDataset as TrainingDatasetRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about a built dataset."""
    total_samples: int = 0
    num_projects: int = 0
    num_trades: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingDatasetConfig:
    """Configuration for dataset building."""
    seq_len: int = 12  # 12 months lookback
    forecast_horizon: int = 1  # Predict 1 month ahead
    min_data_quality: float = 0.6  # Minimum project quality score
    validation_months: int = 6  # Last N months for validation
    batch_size: int = 64
    shuffle_buffer: int = 1000


class TrainingDatasetService:
    """
    Service for building ML training datasets from canonical cost features.

    Key responsibilities:
    - Create sequences with proper temporal alignment
    - Apply time-based splitting to prevent data leakage
    - Handle project and trade ID encoding
    - Normalize features
    """

    # Feature names in the sequence
    FEATURE_NAMES = [
        'cost_per_sf_cents',
        'cumulative_cost_per_sf_cents',
        'budget_per_sf_cents',
        'pct_complete',
        'schedule_pct_elapsed',
    ]

    def __init__(self, db: Session):
        self.db = db
        self._project_id_map: Dict[int, int] = {}
        self._trade_id_map: Dict[int, int] = {}
        self._feature_stats: Dict[str, Tuple[float, float]] = {}

    def build_global_dataset(
        self,
        config: Optional[TrainingDatasetConfig] = None,
    ) -> Tuple['tf.data.Dataset', 'tf.data.Dataset', DatasetStats]:
        """
        Build training and validation datasets from all eligible projects.

        Uses time-based splitting to prevent data leakage:
        - Training: All data before cutoff date
        - Validation: All data after cutoff date

        Args:
            config: Dataset configuration

        Returns:
            Tuple of (train_dataset, val_dataset, stats)
        """
        import tensorflow as tf

        config = config or TrainingDatasetConfig()

        # Get eligible projects
        projects = self.db.query(Project).filter(
            Project.is_training_eligible == True,
            Project.data_quality_score >= config.min_data_quality,
        ).all()

        if not projects:
            logger.warning("No eligible projects found for training")
            return None, None, DatasetStats()

        project_ids = [p.id for p in projects]
        logger.info(f"Building dataset from {len(project_ids)} eligible projects")

        # Build ID mappings (consistent encoding)
        self._build_id_mappings(project_ids)

        # Determine date ranges
        date_range = self._get_date_range(project_ids)
        if not date_range[0] or not date_range[1]:
            logger.warning("No feature data found")
            return None, None, DatasetStats()

        # Time-based split
        cutoff_date = date_range[1] - timedelta(days=config.validation_months * 30)

        # Build sequences for each split
        train_data = self._build_sequences(
            project_ids, config.seq_len, config.forecast_horizon,
            end_date=cutoff_date
        )
        val_data = self._build_sequences(
            project_ids, config.seq_len, config.forecast_horizon,
            start_date=cutoff_date
        )

        if train_data is None or len(train_data['sequences']) == 0:
            logger.warning("No training sequences generated")
            return None, None, DatasetStats()

        # Compute feature statistics from training data only
        self._compute_feature_stats(train_data['sequences'])

        # Normalize sequences
        train_sequences = self._normalize_sequences(train_data['sequences'])
        val_sequences = self._normalize_sequences(val_data['sequences']) if val_data else None

        # Build TensorFlow datasets
        train_dataset = self._to_tf_dataset(
            train_sequences,
            train_data['project_ids'],
            train_data['trade_ids'],
            train_data['targets'],
            config.batch_size,
            config.shuffle_buffer,
            shuffle=True
        )

        val_dataset = None
        if val_sequences is not None and len(val_sequences) > 0:
            val_dataset = self._to_tf_dataset(
                val_sequences,
                val_data['project_ids'],
                val_data['trade_ids'],
                val_data['targets'],
                config.batch_size,
                shuffle=False
            )

        # Build stats
        stats = DatasetStats(
            total_samples=len(train_data['sequences']) + (len(val_data['sequences']) if val_data else 0),
            num_projects=len(project_ids),
            num_trades=len(self._trade_id_map),
            date_range_start=date_range[0],
            date_range_end=date_range[1],
            feature_means={k: v[0] for k, v in self._feature_stats.items()},
            feature_stds={k: v[1] for k, v in self._feature_stats.items()},
        )

        logger.info(
            f"Built dataset: {len(train_data['sequences'])} train, "
            f"{len(val_data['sequences']) if val_data else 0} val samples"
        )

        return train_dataset, val_dataset, stats

    def build_project_dataset(
        self,
        project_id: int,
        config: Optional[TrainingDatasetConfig] = None,
    ) -> 'tf.data.Dataset':
        """
        Build dataset for a single project (for fine-tuning).

        Args:
            project_id: Project to build dataset for
            config: Dataset configuration

        Returns:
            TensorFlow dataset
        """
        import tensorflow as tf

        config = config or TrainingDatasetConfig()

        # Use recent data only for fine-tuning
        recent_cutoff = date.today() - timedelta(days=365)

        data = self._build_sequences(
            [project_id], config.seq_len, config.forecast_horizon,
            start_date=recent_cutoff
        )

        if not data or not data['sequences']:
            return None

        # Use existing feature stats if available, otherwise compute
        if not self._feature_stats:
            self._compute_feature_stats(data['sequences'])

        sequences = self._normalize_sequences(data['sequences'])

        return self._to_tf_dataset(
            sequences,
            data['project_ids'],
            data['trade_ids'],
            data['targets'],
            config.batch_size,
            shuffle=True
        )

    def _build_id_mappings(self, project_ids: List[int]):
        """Build consistent ID mappings for projects and trades."""
        # Project mapping (0 reserved for unknown)
        self._project_id_map = {pid: idx + 1 for idx, pid in enumerate(sorted(project_ids))}

        # Trade mapping from database
        trades = self.db.query(CanonicalTrade).filter(
            CanonicalTrade.is_active == True
        ).order_by(CanonicalTrade.id).all()

        self._trade_id_map = {t.id: idx + 1 for idx, t in enumerate(trades)}

    def _get_date_range(
        self, project_ids: List[int]
    ) -> Tuple[Optional[date], Optional[date]]:
        """Get the date range of available feature data."""
        result = self.db.query(
            func.min(CanonicalCostFeature.period_date),
            func.max(CanonicalCostFeature.period_date)
        ).filter(
            CanonicalCostFeature.project_id.in_(project_ids)
        ).first()

        return result if result else (None, None)

    def _build_sequences(
        self,
        project_ids: List[int],
        seq_len: int,
        forecast_horizon: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict:
        """
        Build input sequences and targets from feature data.

        Creates sliding window sequences with proper temporal alignment.
        Each sequence predicts the cost at t+forecast_horizon.
        """
        sequences = []
        targets = []
        project_id_list = []
        trade_id_list = []

        for project_id in project_ids:
            # Get features for this project, ordered by trade and date
            query = self.db.query(CanonicalCostFeature).filter(
                CanonicalCostFeature.project_id == project_id
            )
            if start_date:
                query = query.filter(CanonicalCostFeature.period_date >= start_date)
            if end_date:
                query = query.filter(CanonicalCostFeature.period_date <= end_date)

            features = query.order_by(
                CanonicalCostFeature.canonical_trade_id,
                CanonicalCostFeature.period_date
            ).all()

            if not features:
                continue

            # Group by trade
            trade_features: Dict[int, List] = {}
            for f in features:
                if f.canonical_trade_id not in trade_features:
                    trade_features[f.canonical_trade_id] = []
                trade_features[f.canonical_trade_id].append(f)

            # Create sequences for each trade
            for trade_id, trade_data in trade_features.items():
                if len(trade_data) < seq_len + forecast_horizon:
                    continue  # Not enough data

                for i in range(len(trade_data) - seq_len - forecast_horizon + 1):
                    # Input sequence
                    seq = []
                    for j in range(seq_len):
                        f = trade_data[i + j]
                        seq.append([
                            f.cost_per_sf_cents or 0,
                            f.cumulative_cost_per_sf_cents or 0,
                            f.budget_per_sf_cents or 0,
                            f.pct_complete or 0,
                            f.schedule_pct_elapsed or 0,
                        ])
                    sequences.append(seq)

                    # Target: cumulative cost at forecast horizon
                    target_idx = i + seq_len + forecast_horizon - 1
                    targets.append(trade_data[target_idx].cumulative_cost_per_sf_cents or 0)

                    # IDs (mapped)
                    project_id_list.append(self._project_id_map.get(project_id, 0))
                    trade_id_list.append(self._trade_id_map.get(trade_id, 0))

        return {
            'sequences': np.array(sequences, dtype=np.float32) if sequences else np.array([]),
            'targets': np.array(targets, dtype=np.float32) if targets else np.array([]),
            'project_ids': np.array(project_id_list, dtype=np.int32) if project_id_list else np.array([]),
            'trade_ids': np.array(trade_id_list, dtype=np.int32) if trade_id_list else np.array([]),
        }

    def _compute_feature_stats(self, sequences: np.ndarray):
        """Compute mean and std for each feature from training data."""
        if len(sequences) == 0:
            return

        for i, name in enumerate(self.FEATURE_NAMES):
            values = sequences[:, :, i].flatten()
            # Filter out zeros for meaningful statistics
            non_zero = values[values != 0]
            if len(non_zero) > 0:
                self._feature_stats[name] = (float(np.mean(non_zero)), float(np.std(non_zero) + 1e-6))
            else:
                self._feature_stats[name] = (0.0, 1.0)

    def _normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Normalize sequences using computed feature statistics."""
        if len(sequences) == 0:
            return sequences

        normalized = sequences.copy()
        for i, name in enumerate(self.FEATURE_NAMES):
            mean, std = self._feature_stats.get(name, (0.0, 1.0))
            normalized[:, :, i] = (sequences[:, :, i] - mean) / std

        return normalized

    def _to_tf_dataset(
        self,
        sequences: np.ndarray,
        project_ids: np.ndarray,
        trade_ids: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        shuffle_buffer: int = 1000,
        shuffle: bool = True,
    ) -> 'tf.data.Dataset':
        """Convert numpy arrays to TensorFlow dataset."""
        import tensorflow as tf

        # Expand IDs to match expected input shape (batch, 1)
        project_ids = project_ids.reshape(-1, 1)
        trade_ids = trade_ids.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            (sequences, project_ids, trade_ids),
            targets
        ))

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_id_mappings(self) -> Dict[str, Dict[int, int]]:
        """Get the project and trade ID mappings for inference."""
        return {
            'project_id_map': self._project_id_map,
            'trade_id_map': self._trade_id_map,
        }

    def get_feature_stats(self) -> Dict[str, Tuple[float, float]]:
        """Get feature normalization statistics."""
        return self._feature_stats.copy()

    def set_feature_stats(self, stats: Dict[str, Tuple[float, float]]):
        """Set feature normalization statistics (for inference)."""
        self._feature_stats = stats.copy()

    def save_dataset_record(
        self,
        name: str,
        stats: DatasetStats,
        config: TrainingDatasetConfig,
    ) -> int:
        """
        Save dataset metadata to database.

        Args:
            name: Dataset name/version
            stats: Dataset statistics
            config: Configuration used

        Returns:
            Dataset record ID
        """
        import json

        record = TrainingDatasetRecord(
            name=name,
            version=1,
            created_at=date.today(),
            num_projects=stats.num_projects,
            num_samples=stats.total_samples,
            date_range_start=stats.date_range_start,
            date_range_end=stats.date_range_end,
            config_json=json.dumps({
                'seq_len': config.seq_len,
                'forecast_horizon': config.forecast_horizon,
                'min_data_quality': config.min_data_quality,
                'validation_months': config.validation_months,
                'feature_stats': {k: list(v) for k, v in self._feature_stats.items()},
            }),
        )
        self.db.add(record)
        self.db.flush()

        logger.info(f"Saved dataset record: {name} (ID: {record.id})")

        return record.id

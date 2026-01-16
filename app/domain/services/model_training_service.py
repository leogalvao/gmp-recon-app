"""
Model Training Service - Orchestrates ML model training and versioning.

Handles:
1. Global foundation model training
2. Project-specific fine-tuning
3. Model checkpointing and versioning
4. Training metrics tracking
"""
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import (
    Project,
    MLModelRegistry,
)
from .training_dataset_service import (
    TrainingDatasetService,
    TrainingDatasetConfig,
    DatasetStats,
)

logger = logging.getLogger(__name__)

# Default model storage path
MODEL_STORAGE_PATH = Path("models")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Dataset config
    seq_len: int = 12
    forecast_horizon: int = 1
    min_data_quality: float = 0.6
    validation_months: int = 6
    batch_size: int = 64

    # Model architecture
    project_embed_dim: int = 32
    trade_embed_dim: int = 16
    lstm_units: int = 64
    adapter_units: int = 32
    dropout: float = 0.2

    # Training hyperparameters
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10

    # Fine-tuning
    finetune_epochs: int = 20
    finetune_learning_rate: float = 1e-4


@dataclass
class TrainingResult:
    """Result of a training run."""
    model_id: int
    model_version: str
    success: bool
    epochs_trained: int
    final_train_loss: float
    final_val_loss: Optional[float]
    training_history: Dict[str, List[float]]
    dataset_stats: DatasetStats
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ModelTrainingService:
    """
    Service for training and managing ML models.

    Orchestrates:
    - Global model training on all historical projects
    - Project-specific fine-tuning
    - Model versioning and registry
    - Checkpoint management
    """

    def __init__(self, db: Session, model_path: Optional[Path] = None):
        self.db = db
        self.model_path = model_path or MODEL_STORAGE_PATH
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train_global_model(
        self,
        config: Optional[TrainingConfig] = None,
        model_name: str = "multi_project_forecaster",
    ) -> TrainingResult:
        """
        Train the global foundation model on all eligible projects.

        Args:
            config: Training configuration
            model_name: Name for the model version

        Returns:
            TrainingResult with metrics and model reference
        """
        # Lazy import to avoid TensorFlow loading at module import
        from app.forecasting.models import get_multi_project_forecaster

        config = config or TrainingConfig()
        errors = []
        warnings = []

        logger.info("Starting global model training")

        # Build dataset
        dataset_service = TrainingDatasetService(self.db)
        dataset_config = TrainingDatasetConfig(
            seq_len=config.seq_len,
            forecast_horizon=config.forecast_horizon,
            min_data_quality=config.min_data_quality,
            validation_months=config.validation_months,
            batch_size=config.batch_size,
        )

        train_dataset, val_dataset, dataset_stats = dataset_service.build_global_dataset(
            config=dataset_config
        )

        if train_dataset is None:
            return TrainingResult(
                model_id=0,
                model_version="",
                success=False,
                epochs_trained=0,
                final_train_loss=float('inf'),
                final_val_loss=None,
                training_history={},
                dataset_stats=DatasetStats(),
                errors=["No training data available"],
            )

        # Get number of projects and trades from mappings
        id_mappings = dataset_service.get_id_mappings()
        num_projects = len(id_mappings['project_id_map'])
        num_trades = len(id_mappings['trade_id_map'])

        logger.info(f"Training with {num_projects} projects, {num_trades} trades")

        # Create model
        MultiProjectForecaster = get_multi_project_forecaster()
        model = MultiProjectForecaster(
            num_projects=num_projects,
            num_trades=num_trades,
            seq_len=config.seq_len,
            feature_dim=len(TrainingDatasetService.FEATURE_NAMES),
            project_embed_dim=config.project_embed_dim,
            trade_embed_dim=config.trade_embed_dim,
            lstm_units=config.lstm_units,
            adapter_units=config.adapter_units,
            dropout=config.dropout,
        )

        # Build model
        model.build_model()
        logger.info(f"Model created with {model.count_params()} parameters")

        # Train
        try:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.model_path / model_name / version / "checkpoints"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            history = model.train_global(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                early_stopping_patience=config.early_stopping_patience,
                checkpoint_path=str(checkpoint_path / "best_model"),
            )

            epochs_trained = len(history['loss'])
            final_train_loss = history['loss'][-1]
            final_val_loss = history.get('val_loss', [None])[-1]

            logger.info(
                f"Training complete: {epochs_trained} epochs, "
                f"train_loss={final_train_loss:.4f}, val_loss={final_val_loss:.4f if final_val_loss else 'N/A'}"
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                model_id=0,
                model_version="",
                success=False,
                epochs_trained=0,
                final_train_loss=float('inf'),
                final_val_loss=None,
                training_history={},
                dataset_stats=dataset_stats,
                errors=[str(e)],
            )

        # Save model
        model_dir = self.model_path / model_name / version
        model.save(str(model_dir))

        # Save feature stats and ID mappings
        metadata = {
            'id_mappings': id_mappings,
            'feature_stats': dataset_service.get_feature_stats(),
            'config': {
                'seq_len': config.seq_len,
                'forecast_horizon': config.forecast_horizon,
                'project_embed_dim': config.project_embed_dim,
                'trade_embed_dim': config.trade_embed_dim,
                'lstm_units': config.lstm_units,
                'adapter_units': config.adapter_units,
            },
        }
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Register model in database
        model_record = MLModelRegistry(
            name=model_name,
            version=version,
            model_type='multi_project_forecaster',
            created_at=datetime.now(),
            training_dataset_id=None,  # Could link to saved dataset record
            metrics_json=json.dumps({
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'epochs_trained': epochs_trained,
                'num_projects': num_projects,
                'num_trades': num_trades,
            }),
            model_path=str(model_dir),
            is_active=True,
        )
        self.db.add(model_record)
        self.db.flush()

        # Deactivate previous versions
        self.db.query(MLModelRegistry).filter(
            MLModelRegistry.name == model_name,
            MLModelRegistry.id != model_record.id,
        ).update({'is_active': False})

        self.db.commit()

        logger.info(f"Model registered: {model_name} v{version} (ID: {model_record.id})")

        return TrainingResult(
            model_id=model_record.id,
            model_version=version,
            success=True,
            epochs_trained=epochs_trained,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            training_history=history,
            dataset_stats=dataset_stats,
            warnings=warnings,
        )

    def finetune_for_project(
        self,
        project_id: int,
        base_model_id: Optional[int] = None,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingResult:
        """
        Fine-tune the global model for a specific project.

        Args:
            project_id: Project to fine-tune for
            base_model_id: ID of base model (default: latest active)
            config: Training configuration

        Returns:
            TrainingResult with fine-tuning metrics
        """
        from app.forecasting.models import get_multi_project_forecaster

        config = config or TrainingConfig()

        # Get base model
        if base_model_id:
            base_model_record = self.db.query(MLModelRegistry).get(base_model_id)
        else:
            base_model_record = self.db.query(MLModelRegistry).filter(
                MLModelRegistry.is_active == True,
                MLModelRegistry.model_type == 'multi_project_forecaster',
            ).first()

        if not base_model_record:
            return TrainingResult(
                model_id=0,
                model_version="",
                success=False,
                epochs_trained=0,
                final_train_loss=float('inf'),
                final_val_loss=None,
                training_history={},
                dataset_stats=DatasetStats(),
                errors=["No base model found"],
            )

        # Load metadata
        model_dir = Path(base_model_record.model_path)
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Build project dataset
        dataset_service = TrainingDatasetService(self.db)
        dataset_service.set_feature_stats(metadata['feature_stats'])
        dataset_service._project_id_map = metadata['id_mappings']['project_id_map']
        dataset_service._trade_id_map = metadata['id_mappings']['trade_id_map']

        project_dataset = dataset_service.build_project_dataset(
            project_id=project_id,
            config=TrainingDatasetConfig(
                seq_len=config.seq_len,
                forecast_horizon=config.forecast_horizon,
                batch_size=config.batch_size,
            ),
        )

        if project_dataset is None:
            return TrainingResult(
                model_id=0,
                model_version="",
                success=False,
                epochs_trained=0,
                final_train_loss=float('inf'),
                final_val_loss=None,
                training_history={},
                dataset_stats=DatasetStats(),
                errors=[f"No data available for project {project_id}"],
            )

        # Load model
        MultiProjectForecaster = get_multi_project_forecaster()
        model = MultiProjectForecaster(
            num_projects=len(metadata['id_mappings']['project_id_map']),
            num_trades=len(metadata['id_mappings']['trade_id_map']),
            **metadata['config'],
        )
        model.load(str(model_dir))

        # Fine-tune
        try:
            history = model.finetune_for_project(
                project_id=metadata['id_mappings']['project_id_map'].get(project_id, 0),
                project_dataset=project_dataset,
                epochs=config.finetune_epochs,
                learning_rate=config.finetune_learning_rate,
            )

            epochs_trained = len(history['loss'])
            final_train_loss = history['loss'][-1]

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return TrainingResult(
                model_id=0,
                model_version="",
                success=False,
                epochs_trained=0,
                final_train_loss=float('inf'),
                final_val_loss=None,
                training_history={},
                dataset_stats=DatasetStats(),
                errors=[str(e)],
            )

        # Save fine-tuned adapter
        version = f"{base_model_record.version}_ft_{project_id}"
        adapter_dir = self.model_path / base_model_record.name / version
        model.save(str(adapter_dir))

        # Copy metadata with project-specific info
        metadata['finetuned_project_id'] = project_id
        with open(adapter_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Register in database
        adapter_record = MLModelRegistry(
            name=f"{base_model_record.name}_project_{project_id}",
            version=version,
            model_type='project_adapter',
            created_at=datetime.now(),
            training_dataset_id=None,
            metrics_json=json.dumps({
                'final_train_loss': final_train_loss,
                'epochs_trained': epochs_trained,
                'base_model_id': base_model_record.id,
                'project_id': project_id,
            }),
            model_path=str(adapter_dir),
            is_active=True,
        )
        self.db.add(adapter_record)
        self.db.commit()

        logger.info(f"Fine-tuned model saved for project {project_id}")

        return TrainingResult(
            model_id=adapter_record.id,
            model_version=version,
            success=True,
            epochs_trained=epochs_trained,
            final_train_loss=final_train_loss,
            final_val_loss=None,
            training_history=history,
            dataset_stats=DatasetStats(num_projects=1),
        )

    def get_active_model(
        self,
        model_name: str = "multi_project_forecaster"
    ) -> Optional[MLModelRegistry]:
        """Get the currently active model version."""
        return self.db.query(MLModelRegistry).filter(
            MLModelRegistry.name == model_name,
            MLModelRegistry.is_active == True,
        ).first()

    def list_models(
        self,
        model_type: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[MLModelRegistry]:
        """List registered models."""
        query = self.db.query(MLModelRegistry)

        if model_type:
            query = query.filter(MLModelRegistry.model_type == model_type)

        if not include_inactive:
            query = query.filter(MLModelRegistry.is_active == True)

        return query.order_by(MLModelRegistry.created_at.desc()).all()

    def set_active_model(self, model_id: int) -> bool:
        """Set a specific model version as active."""
        model = self.db.query(MLModelRegistry).get(model_id)
        if not model:
            return False

        # Deactivate others with same name
        self.db.query(MLModelRegistry).filter(
            MLModelRegistry.name == model.name,
        ).update({'is_active': False})

        # Activate this one
        model.is_active = True
        self.db.commit()

        logger.info(f"Set model {model_id} as active")
        return True

"""
ML Pipeline API Endpoints - Model training, evaluation, and forecasting.

Implements Phase 3 ML operations:
- POST /api/v1/ml/train - Train global model
- POST /api/v1/ml/finetune/{project_id} - Fine-tune for project
- GET /api/v1/ml/models - List registered models
- GET /api/v1/ml/models/{model_id} - Get model details
- POST /api/v1/ml/models/{model_id}/activate - Set model as active
- POST /api/v1/ml/evaluate/{model_id} - Evaluate model
- GET /api/v1/ml/forecast/{project_id} - Get project forecast
- POST /api/v1/ml/forecast/batch - Generate batch forecasts
"""
from typing import List, Optional, Dict, Any
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db, User, MLModelRegistry
from app.domain.services import (
    ModelTrainingService,
    TrainingConfig,
    ModelEvaluationService,
    ForecastInferenceService,
)
from app.api.v1.auth import get_current_active_user

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class TrainingConfigRequest(BaseModel):
    """Request model for training configuration."""
    seq_len: int = Field(12, ge=3, le=24, description="Sequence length (months)")
    forecast_horizon: int = Field(1, ge=1, le=6, description="Forecast horizon (months)")
    min_data_quality: float = Field(0.6, ge=0.0, le=1.0, description="Min project quality")
    validation_months: int = Field(6, ge=1, le=12, description="Validation period")
    batch_size: int = Field(64, ge=16, le=256, description="Training batch size")
    epochs: int = Field(100, ge=10, le=500, description="Max training epochs")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate")
    early_stopping_patience: int = Field(10, ge=3, le=50, description="Early stopping patience")


class TrainModelRequest(BaseModel):
    """Request model for training a new model."""
    model_name: str = Field("multi_project_forecaster", description="Model name")
    config: Optional[TrainingConfigRequest] = None


class TrainingResultResponse(BaseModel):
    """Response model for training result."""
    model_id: int
    model_version: str
    success: bool
    epochs_trained: int
    final_train_loss: float
    final_val_loss: Optional[float]
    num_projects: int
    num_trades: int
    errors: List[str]
    warnings: List[str]


class FinetuneRequest(BaseModel):
    """Request model for fine-tuning."""
    base_model_id: Optional[int] = Field(None, description="Base model (default: active)")
    epochs: int = Field(20, ge=5, le=100, description="Fine-tuning epochs")
    learning_rate: float = Field(0.0001, ge=0.00001, le=0.01, description="Learning rate")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    id: int
    name: str
    version: str
    model_type: str
    created_at: str
    is_active: bool
    metrics: Optional[Dict[str, Any]]
    model_path: str


class EvaluationMetricsResponse(BaseModel):
    """Response model for evaluation metrics."""
    mape: float
    mae: float
    rmse: float
    r2: float
    coverage_80: float
    coverage_90: float
    avg_interval_width: float
    num_samples: int


class ProjectEvaluationResponse(BaseModel):
    """Response model for project evaluation."""
    project_id: int
    project_code: str
    metrics: EvaluationMetricsResponse


class EvaluationResultResponse(BaseModel):
    """Response model for model evaluation."""
    model_id: int
    overall_metrics: EvaluationMetricsResponse
    project_evaluations: List[ProjectEvaluationResponse]


class TradeForecastResponse(BaseModel):
    """Response model for trade forecast."""
    canonical_trade_id: int
    canonical_code: str
    canonical_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    current_cumulative_cost: float
    budget: float
    forecasted_eac: float


class ProjectForecastResponse(BaseModel):
    """Response model for project forecast."""
    project_id: int
    project_code: str
    as_of_date: str
    total_budget: float
    total_cumulative_cost: float
    total_forecasted_eac: float
    total_forecast_lower: float
    total_forecast_upper: float
    confidence_level: float
    trade_forecasts: List[TradeForecastResponse]
    model_version: str


class BatchForecastRequest(BaseModel):
    """Request model for batch forecasting."""
    project_ids: Optional[List[int]] = Field(None, description="Projects to forecast (default: all)")
    model_id: Optional[int] = Field(None, description="Model to use (default: active)")
    save_results: bool = Field(True, description="Save forecasts to database")


class BatchForecastResponse(BaseModel):
    """Response model for batch forecasting."""
    total_projects: int
    successful: int
    failed: int
    forecasts: List[ProjectForecastResponse]


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/train",
    response_model=TrainingResultResponse,
    summary="Train global model",
    description="Train a new global foundation model on all eligible projects"
)
def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Train a new global foundation model.

    This is a long-running operation that trains on all historical project data.
    """
    service = ModelTrainingService(db)

    # Build config
    config = TrainingConfig()
    if request.config:
        config.seq_len = request.config.seq_len
        config.forecast_horizon = request.config.forecast_horizon
        config.min_data_quality = request.config.min_data_quality
        config.validation_months = request.config.validation_months
        config.batch_size = request.config.batch_size
        config.epochs = request.config.epochs
        config.learning_rate = request.config.learning_rate
        config.early_stopping_patience = request.config.early_stopping_patience

    # Train model
    result = service.train_global_model(
        config=config,
        model_name=request.model_name,
    )

    # Extract metrics from result
    import json
    model_record = db.query(MLModelRegistry).get(result.model_id) if result.model_id else None
    metrics = json.loads(model_record.metrics_json) if model_record and model_record.metrics_json else {}

    return {
        'model_id': result.model_id,
        'model_version': result.model_version,
        'success': result.success,
        'epochs_trained': result.epochs_trained,
        'final_train_loss': result.final_train_loss,
        'final_val_loss': result.final_val_loss,
        'num_projects': metrics.get('num_projects', result.dataset_stats.num_projects),
        'num_trades': metrics.get('num_trades', result.dataset_stats.num_trades),
        'errors': result.errors,
        'warnings': result.warnings,
    }


@router.post(
    "/finetune/{project_id}",
    response_model=TrainingResultResponse,
    summary="Fine-tune for project",
    description="Fine-tune the global model for a specific project"
)
def finetune_for_project(
    project_id: int,
    request: FinetuneRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Fine-tune the model adapter for a specific project."""
    service = ModelTrainingService(db)

    config = TrainingConfig(
        finetune_epochs=request.epochs,
        finetune_learning_rate=request.learning_rate,
    )

    result = service.finetune_for_project(
        project_id=project_id,
        base_model_id=request.base_model_id,
        config=config,
    )

    return {
        'model_id': result.model_id,
        'model_version': result.model_version,
        'success': result.success,
        'epochs_trained': result.epochs_trained,
        'final_train_loss': result.final_train_loss,
        'final_val_loss': result.final_val_loss,
        'num_projects': 1,
        'num_trades': 0,
        'errors': result.errors,
        'warnings': result.warnings,
    }


@router.get(
    "/models",
    response_model=List[ModelInfoResponse],
    summary="List models",
    description="List all registered ML models"
)
def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    include_inactive: bool = Query(False, description="Include inactive models"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all registered models."""
    service = ModelTrainingService(db)
    models = service.list_models(model_type, include_inactive)

    import json
    return [
        {
            'id': m.id,
            'name': m.model_name,
            'version': m.model_version,
            'model_type': m.model_type,
            'created_at': m.created_at.isoformat() if m.created_at else None,
            'is_active': m.is_production,
            'metrics': json.loads(m.metrics) if m.metrics else None,
            'model_path': m.artifact_path,
        }
        for m in models
    ]


@router.get(
    "/models/{model_id}",
    response_model=ModelInfoResponse,
    summary="Get model details",
    description="Get detailed information about a specific model"
)
def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get model details."""
    model = db.query(MLModelRegistry).get(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    import json
    return {
        'id': model.id,
        'name': model.model_name,
        'version': model.model_version,
        'model_type': model.model_type,
        'created_at': model.created_at.isoformat() if model.created_at else None,
        'is_active': model.is_production,
        'metrics': json.loads(model.metrics) if model.metrics else None,
        'model_path': model.artifact_path,
    }


@router.post(
    "/models/{model_id}/activate",
    summary="Activate model",
    description="Set a specific model version as the active model"
)
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Set a model as the active version."""
    service = ModelTrainingService(db)
    success = service.set_active_model(model_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    return {'message': f'Model {model_id} activated successfully'}


@router.post(
    "/evaluate/{model_id}",
    response_model=EvaluationResultResponse,
    summary="Evaluate model",
    description="Evaluate model performance on test data"
)
def evaluate_model(
    model_id: int,
    test_start_date: Optional[date] = Query(None, description="Test period start"),
    test_end_date: Optional[date] = Query(None, description="Test period end"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Evaluate a model on held-out test data."""
    service = ModelEvaluationService(db)

    overall_metrics, project_evaluations = service.evaluate_model(
        model_id=model_id,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )

    return {
        'model_id': model_id,
        'overall_metrics': {
            'mape': overall_metrics.mape,
            'mae': overall_metrics.mae,
            'rmse': overall_metrics.rmse,
            'r2': overall_metrics.r2,
            'coverage_80': overall_metrics.coverage_80,
            'coverage_90': overall_metrics.coverage_90,
            'avg_interval_width': overall_metrics.avg_interval_width,
            'num_samples': overall_metrics.num_samples,
        },
        'project_evaluations': [
            {
                'project_id': pe.project_id,
                'project_code': pe.project_code,
                'metrics': {
                    'mape': pe.metrics.mape,
                    'mae': pe.metrics.mae,
                    'rmse': pe.metrics.rmse,
                    'r2': pe.metrics.r2,
                    'coverage_80': pe.metrics.coverage_80,
                    'coverage_90': pe.metrics.coverage_90,
                    'avg_interval_width': pe.metrics.avg_interval_width,
                    'num_samples': pe.metrics.num_samples,
                }
            }
            for pe in project_evaluations
        ]
    }


@router.get(
    "/forecast/{project_id}",
    response_model=ProjectForecastResponse,
    summary="Get project forecast",
    description="Generate cost forecast for a project"
)
def get_project_forecast(
    project_id: int,
    model_id: Optional[int] = Query(None, description="Model to use (default: active)"),
    as_of_date: Optional[date] = Query(None, description="Forecast as-of date"),
    confidence_level: float = Query(0.80, ge=0.5, le=0.99, description="Confidence level"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate forecast for a project."""
    service = ForecastInferenceService(db)

    forecast = service.get_project_forecast(
        project_id=project_id,
        model_id=model_id,
        as_of_date=as_of_date,
        confidence_level=confidence_level,
    )

    if not forecast:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to generate forecast for project {project_id}"
        )

    return {
        'project_id': forecast.project_id,
        'project_code': forecast.project_code,
        'as_of_date': forecast.as_of_date.isoformat(),
        'total_budget': forecast.total_budget,
        'total_cumulative_cost': forecast.total_cumulative_cost,
        'total_forecasted_eac': forecast.total_forecasted_eac,
        'total_forecast_lower': forecast.total_forecast_lower,
        'total_forecast_upper': forecast.total_forecast_upper,
        'confidence_level': forecast.confidence_level,
        'trade_forecasts': [
            {
                'canonical_trade_id': tf.canonical_trade_id,
                'canonical_code': tf.canonical_code,
                'canonical_name': tf.canonical_name,
                'point_estimate': tf.forecast.point_estimate,
                'lower_bound': tf.forecast.lower_bound,
                'upper_bound': tf.forecast.upper_bound,
                'confidence_level': tf.forecast.confidence_level,
                'current_cumulative_cost': tf.current_cumulative_cost,
                'budget': tf.budget,
                'forecasted_eac': tf.forecasted_eac,
            }
            for tf in forecast.trade_forecasts
        ],
        'model_version': forecast.model_version,
    }


@router.post(
    "/forecast/batch",
    response_model=BatchForecastResponse,
    summary="Batch forecast",
    description="Generate forecasts for multiple projects"
)
def batch_forecast(
    request: BatchForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate forecasts for multiple projects."""
    from app.domain.services.forecast_inference_service import generate_batch_forecasts

    results = generate_batch_forecasts(
        db=db,
        project_ids=request.project_ids,
        model_id=request.model_id,
    )

    forecasts = []
    for project_id, forecast in results.items():
        forecasts.append({
            'project_id': forecast.project_id,
            'project_code': forecast.project_code,
            'as_of_date': forecast.as_of_date.isoformat(),
            'total_budget': forecast.total_budget,
            'total_cumulative_cost': forecast.total_cumulative_cost,
            'total_forecasted_eac': forecast.total_forecasted_eac,
            'total_forecast_lower': forecast.total_forecast_lower,
            'total_forecast_upper': forecast.total_forecast_upper,
            'confidence_level': forecast.confidence_level,
            'trade_forecasts': [
                {
                    'canonical_trade_id': tf.canonical_trade_id,
                    'canonical_code': tf.canonical_code,
                    'canonical_name': tf.canonical_name,
                    'point_estimate': tf.forecast.point_estimate,
                    'lower_bound': tf.forecast.lower_bound,
                    'upper_bound': tf.forecast.upper_bound,
                    'confidence_level': tf.forecast.confidence_level,
                    'current_cumulative_cost': tf.current_cumulative_cost,
                    'budget': tf.budget,
                    'forecasted_eac': tf.forecasted_eac,
                }
                for tf in forecast.trade_forecasts
            ],
            'model_version': forecast.model_version,
        })

    return {
        'total_projects': len(request.project_ids) if request.project_ids else len(results),
        'successful': len(results),
        'failed': (len(request.project_ids) if request.project_ids else 0) - len(results),
        'forecasts': forecasts,
    }


@router.get(
    "/calibration/{model_id}",
    summary="Assess calibration",
    description="Assess probabilistic calibration of a model"
)
def assess_calibration(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Assess the probabilistic calibration of a model."""
    service = ModelEvaluationService(db)
    calibration = service.assess_calibration(model_id)

    return {
        'model_id': model_id,
        'calibration': calibration,
        'interpretation': {
            level: 'well-calibrated' if abs(coverage - level) < 0.05
            else 'overconfident' if coverage < level
            else 'underconfident'
            for level, coverage in calibration.items()
        }
    }

"""
Forecast Inference Service - Real-time cost predictions using trained models.

Handles:
1. Loading and caching trained models
2. Feature preparation for inference
3. Generating forecasts with uncertainty
4. Storing forecast results
"""
import logging
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from sqlalchemy.orm import Session

from app.models import (
    Project,
    GMP,
    CanonicalCostFeature,
    CanonicalTrade,
    MLModelRegistry,
    ProjectForecast,
)
from app.forecasting.models.base_model import ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class TradeForecast:
    """Forecast result for a single trade."""
    canonical_trade_id: int
    canonical_code: str
    canonical_name: str
    forecast: ForecastResult
    current_cumulative_cost: float
    budget: float
    forecasted_eac: float  # Estimated At Completion


@dataclass
class ProjectForecastResult:
    """Aggregated forecast result for a project."""
    project_id: int
    project_code: str
    as_of_date: date
    total_budget: float
    total_cumulative_cost: float
    total_forecasted_eac: float
    total_forecast_lower: float
    total_forecast_upper: float
    confidence_level: float
    trade_forecasts: List[TradeForecast]
    model_version: str


class ForecastInferenceService:
    """
    Service for generating cost forecasts using trained ML models.

    Provides:
    - Real-time forecast generation for projects/trades
    - Model caching for efficient inference
    - Forecast storage and retrieval
    """

    def __init__(self, db: Session):
        self.db = db
        self._model_cache: Dict[int, Tuple] = {}  # model_id -> (model, metadata)

    def get_project_forecast(
        self,
        project_id: int,
        model_id: Optional[int] = None,
        as_of_date: Optional[date] = None,
        confidence_level: float = 0.80,
    ) -> Optional[ProjectForecastResult]:
        """
        Generate forecasts for all trades in a project.

        Args:
            project_id: Project to forecast
            model_id: Model to use (default: active model)
            as_of_date: Date for forecast (default: today)
            confidence_level: Confidence interval width

        Returns:
            ProjectForecastResult with per-trade and aggregated forecasts
        """
        if not as_of_date:
            as_of_date = date.today()

        # Get project
        project = self.db.query(Project).get(project_id)
        if not project:
            logger.error(f"Project {project_id} not found")
            return None

        # Load model
        model, metadata = self._load_model(model_id)
        if model is None:
            logger.error("No model available for inference")
            return None

        # Get project's GMPs with canonical trades
        gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id != None
        ).all()

        if not gmps:
            logger.warning(f"No GMPs with canonical trades for project {project_id}")
            return None

        # Generate forecasts per trade
        trade_forecasts = []
        total_budget = 0
        total_cumulative = 0
        total_eac = 0
        total_lower = 0
        total_upper = 0

        for gmp in gmps:
            trade_forecast = self._forecast_trade(
                model, metadata, project_id, gmp,
                as_of_date, confidence_level
            )

            if trade_forecast:
                trade_forecasts.append(trade_forecast)
                total_budget += trade_forecast.budget
                total_cumulative += trade_forecast.current_cumulative_cost
                total_eac += trade_forecast.forecasted_eac
                total_lower += trade_forecast.forecast.lower_bound
                total_upper += trade_forecast.forecast.upper_bound

        if not trade_forecasts:
            return None

        # Get model version
        model_record = self.db.query(MLModelRegistry).get(
            model_id or self._get_active_model_id()
        )
        model_version = model_record.model_version if model_record else "unknown"

        return ProjectForecastResult(
            project_id=project_id,
            project_code=project.code,
            as_of_date=as_of_date,
            total_budget=total_budget,
            total_cumulative_cost=total_cumulative,
            total_forecasted_eac=total_eac,
            total_forecast_lower=total_lower,
            total_forecast_upper=total_upper,
            confidence_level=confidence_level,
            trade_forecasts=trade_forecasts,
            model_version=model_version,
        )

    def _forecast_trade(
        self,
        model,
        metadata: Dict,
        project_id: int,
        gmp: GMP,
        as_of_date: date,
        confidence_level: float,
    ) -> Optional[TradeForecast]:
        """Generate forecast for a single trade/GMP."""
        trade_id = gmp.canonical_trade_id
        seq_len = metadata['config']['seq_len']

        # Get feature stats for normalization
        feature_stats = metadata['feature_stats']

        # Get historical features for this project/trade
        features = self.db.query(CanonicalCostFeature).filter(
            CanonicalCostFeature.project_id == project_id,
            CanonicalCostFeature.canonical_trade_id == trade_id,
            CanonicalCostFeature.period_date <= as_of_date,
        ).order_by(
            CanonicalCostFeature.period_date.desc()
        ).limit(seq_len).all()

        if len(features) < seq_len:
            logger.debug(
                f"Insufficient history for project {project_id}, trade {trade_id}: "
                f"{len(features)}/{seq_len} periods"
            )
            return None

        # Build sequence (reverse to chronological order)
        features = features[::-1]
        seq_data = []
        for f in features:
            seq_data.append([
                f.cost_per_sf_cents or 0,
                f.cumulative_cost_per_sf_cents or 0,
                f.budget_per_sf_cents or 0,
                f.pct_complete or 0,
                f.schedule_pct_elapsed or 0,
            ])

        # Normalize
        seq_array = np.array(seq_data, dtype=np.float32)
        feature_names = [
            'cost_per_sf_cents',
            'cumulative_cost_per_sf_cents',
            'budget_per_sf_cents',
            'pct_complete',
            'schedule_pct_elapsed',
        ]
        for i, name in enumerate(feature_names):
            mean, std = feature_stats.get(name, (0, 1))
            seq_array[:, i] = (seq_array[:, i] - mean) / std

        # Get mapped IDs
        mapped_project = metadata['id_mappings']['project_id_map'].get(str(project_id), 0)
        mapped_trade = metadata['id_mappings']['trade_id_map'].get(str(trade_id), 0)

        # Generate forecast
        try:
            forecast = model.predict_with_uncertainty(
                seq_array,
                project_id=mapped_project,
                trade_id=mapped_trade,
                confidence_level=confidence_level,
            )
        except Exception as e:
            logger.error(f"Forecast failed for project {project_id}, trade {trade_id}: {e}")
            return None

        # Get trade info
        trade = self.db.query(CanonicalTrade).get(trade_id)

        # Current values (unnormalized - from original feature data)
        current_cumulative = features[-1].cumulative_cost_per_sf_cents or 0
        budget_per_sf = features[-1].budget_per_sf_cents or 0

        # Denormalize forecast (it predicts cumulative_cost_per_sf)
        cum_mean, cum_std = feature_stats.get('cumulative_cost_per_sf_cents', (0, 1))
        forecast_mean_denorm = forecast.mean * cum_std + cum_mean
        forecast_std_denorm = forecast.std * cum_std

        # Scale back to total cost using project's square footage
        project = self.db.query(Project).get(project_id)
        sf = project.total_square_feet or 1

        forecasted_eac = (forecast_mean_denorm / 100) * sf  # cents to dollars * sf
        budget = (budget_per_sf / 100) * sf

        return TradeForecast(
            canonical_trade_id=trade_id,
            canonical_code=trade.canonical_code if trade else "",
            canonical_name=trade.canonical_name if trade else "",
            forecast=ForecastResult(
                point_estimate=forecast_mean_denorm,
                lower_bound=forecast_mean_denorm - 1.28 * forecast_std_denorm,  # 80% CI
                upper_bound=forecast_mean_denorm + 1.28 * forecast_std_denorm,
                confidence_level=confidence_level,
                mean=forecast_mean_denorm,
                std=forecast_std_denorm,
            ),
            current_cumulative_cost=(current_cumulative / 100) * sf,
            budget=budget,
            forecasted_eac=forecasted_eac,
        )

    def save_forecast(
        self,
        forecast_result: ProjectForecastResult,
    ) -> int:
        """
        Save a forecast result to the database.

        Args:
            forecast_result: Forecast to save

        Returns:
            Forecast record ID
        """
        # Get model ID
        model_record = self.db.query(MLModelRegistry).filter(
            MLModelRegistry.version == forecast_result.model_version
        ).first()

        record = ProjectForecast(
            project_id=forecast_result.project_id,
            model_id=model_record.id if model_record else None,
            forecast_date=forecast_result.as_of_date,
            forecast_horizon_months=1,
            predicted_eac_cents=int(forecast_result.total_forecasted_eac * 100),
            confidence_lower_cents=int(forecast_result.total_forecast_lower * 100),
            confidence_upper_cents=int(forecast_result.total_forecast_upper * 100),
            confidence_level=forecast_result.confidence_level,
            actual_eac_cents=None,  # To be filled when actuals are available
            created_at=date.today(),
        )
        self.db.add(record)
        self.db.flush()

        logger.info(f"Saved forecast {record.id} for project {forecast_result.project_id}")

        return record.id

    def get_historical_forecasts(
        self,
        project_id: int,
        limit: int = 12,
    ) -> List[ProjectForecast]:
        """Get historical forecasts for a project."""
        return self.db.query(ProjectForecast).filter(
            ProjectForecast.project_id == project_id
        ).order_by(
            ProjectForecast.forecast_date.desc()
        ).limit(limit).all()

    def _load_model(
        self,
        model_id: Optional[int] = None,
    ) -> Tuple[Optional[any], Optional[Dict]]:
        """Load model (with caching)."""
        if model_id is None:
            model_id = self._get_active_model_id()

        if model_id is None:
            return None, None

        # Check cache
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        # Load from disk
        from app.forecasting.models import get_multi_project_forecaster

        model_record = self.db.query(MLModelRegistry).get(model_id)
        if not model_record:
            return None, None

        model_dir = Path(model_record.artifact_path)
        if not model_dir.exists():
            logger.error(f"Model path not found: {model_dir}")
            return None, None

        # Load metadata
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Create and load model
        MultiProjectForecaster = get_multi_project_forecaster()
        # Filter config to only include model parameters
        model_params = {
            k: v for k, v in metadata['config'].items()
            if k in ['seq_len', 'feature_dim', 'project_embed_dim', 'trade_embed_dim',
                     'lstm_units', 'adapter_units', 'dropout']
        }
        model = MultiProjectForecaster(
            num_projects=len(metadata['id_mappings']['project_id_map']),
            num_trades=len(metadata['id_mappings']['trade_id_map']),
            **model_params,
        )
        model.load(str(model_dir))

        # Cache
        self._model_cache[model_id] = (model, metadata)

        logger.info(f"Loaded model {model_id} from {model_dir}")

        return model, metadata

    def _get_active_model_id(self) -> Optional[int]:
        """Get the active model ID."""
        # First try production model
        model = self.db.query(MLModelRegistry).filter(
            MLModelRegistry.is_production == True,
            MLModelRegistry.model_type == 'global',
        ).first()

        # Fall back to most recent model if no production model
        if not model:
            model = self.db.query(MLModelRegistry).filter(
                MLModelRegistry.model_type == 'global',
            ).order_by(MLModelRegistry.created_at.desc()).first()

        return model.id if model else None

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")


def generate_batch_forecasts(
    db: Session,
    project_ids: Optional[List[int]] = None,
    model_id: Optional[int] = None,
) -> Dict[int, ProjectForecastResult]:
    """
    Generate forecasts for multiple projects.

    Args:
        db: Database session
        project_ids: Projects to forecast (default: all eligible)
        model_id: Model to use

    Returns:
        Dict mapping project_id to forecast result
    """
    service = ForecastInferenceService(db)

    if project_ids is None:
        # Get all training-eligible projects
        projects = db.query(Project).filter(
            Project.is_training_eligible == True
        ).all()
        project_ids = [p.id for p in projects]

    results = {}
    for project_id in project_ids:
        forecast = service.get_project_forecast(project_id, model_id)
        if forecast:
            results[project_id] = forecast
            service.save_forecast(forecast)

    db.commit()

    logger.info(f"Generated forecasts for {len(results)}/{len(project_ids)} projects")

    return results

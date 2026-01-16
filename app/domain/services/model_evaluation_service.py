"""
Model Evaluation Service - Metrics, backtesting, and model comparison.

Handles:
1. Per-project and per-trade evaluation metrics
2. Time-series backtesting with walk-forward validation
3. Model comparison across versions
4. Calibration assessment for probabilistic outputs
"""
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import (
    Project,
    CanonicalCostFeature,
    CanonicalTrade,
    MLModelRegistry,
    ProjectForecast,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a model or project."""
    mape: float = 0.0  # Mean Absolute Percentage Error
    mae: float = 0.0   # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Squared Error
    mse: float = 0.0   # Mean Squared Error
    r2: float = 0.0    # R-squared
    coverage_80: float = 0.0  # 80% confidence interval coverage
    coverage_90: float = 0.0  # 90% confidence interval coverage
    avg_interval_width: float = 0.0  # Average prediction interval width
    num_samples: int = 0


@dataclass
class ProjectEvaluation:
    """Evaluation results for a single project."""
    project_id: int
    project_code: str
    metrics: EvaluationMetrics
    trade_metrics: Dict[str, EvaluationMetrics] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result of a backtesting run."""
    model_id: int
    backtest_periods: int
    overall_metrics: EvaluationMetrics
    project_evaluations: List[ProjectEvaluation]
    period_metrics: List[Tuple[date, EvaluationMetrics]]


class ModelEvaluationService:
    """
    Service for evaluating ML model performance.

    Provides:
    - Standard regression metrics (MAPE, MAE, RMSE, R²)
    - Probabilistic calibration metrics (coverage, interval width)
    - Per-project and per-trade breakdown
    - Walk-forward backtesting
    """

    def __init__(self, db: Session):
        self.db = db

    def evaluate_model(
        self,
        model_id: int,
        test_start_date: Optional[date] = None,
        test_end_date: Optional[date] = None,
    ) -> Tuple[EvaluationMetrics, List[ProjectEvaluation]]:
        """
        Evaluate a model on held-out test data.

        Args:
            model_id: Model to evaluate
            test_start_date: Start of test period
            test_end_date: End of test period

        Returns:
            Tuple of (overall_metrics, project_evaluations)
        """
        import json
        from pathlib import Path
        from app.forecasting.models import get_multi_project_forecaster
        from .training_dataset_service import TrainingDatasetService

        # Load model
        model_record = self.db.query(MLModelRegistry).get(model_id)
        if not model_record:
            logger.error(f"Model {model_id} not found")
            return EvaluationMetrics(), []

        model_dir = Path(model_record.model_path)
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Create and load model
        MultiProjectForecaster = get_multi_project_forecaster()
        model = MultiProjectForecaster(
            num_projects=len(metadata['id_mappings']['project_id_map']),
            num_trades=len(metadata['id_mappings']['trade_id_map']),
            **metadata['config'],
        )
        model.load(str(model_dir))

        # Set up dataset service with saved stats
        dataset_service = TrainingDatasetService(self.db)
        dataset_service.set_feature_stats(metadata['feature_stats'])
        dataset_service._project_id_map = metadata['id_mappings']['project_id_map']
        dataset_service._trade_id_map = metadata['id_mappings']['trade_id_map']

        # Default test period: last 3 months
        if not test_end_date:
            test_end_date = date.today()
        if not test_start_date:
            test_start_date = test_end_date - timedelta(days=90)

        # Collect predictions and actuals for each project
        all_predictions = []
        all_actuals = []
        all_stds = []
        project_evaluations = []

        for project_id, mapped_id in metadata['id_mappings']['project_id_map'].items():
            project = self.db.query(Project).get(int(project_id))
            if not project:
                continue

            pred_proj, actual_proj, std_proj = self._evaluate_project(
                model, dataset_service, int(project_id), mapped_id,
                metadata['id_mappings']['trade_id_map'],
                metadata['config']['seq_len'],
                test_start_date, test_end_date
            )

            if len(pred_proj) > 0:
                all_predictions.extend(pred_proj)
                all_actuals.extend(actual_proj)
                all_stds.extend(std_proj)

                # Per-project metrics
                proj_metrics = self._compute_metrics(
                    np.array(pred_proj),
                    np.array(actual_proj),
                    np.array(std_proj)
                )
                project_evaluations.append(ProjectEvaluation(
                    project_id=int(project_id),
                    project_code=project.code,
                    metrics=proj_metrics,
                ))

        # Overall metrics
        if all_predictions:
            overall_metrics = self._compute_metrics(
                np.array(all_predictions),
                np.array(all_actuals),
                np.array(all_stds)
            )
        else:
            overall_metrics = EvaluationMetrics()

        logger.info(
            f"Model {model_id} evaluation: MAPE={overall_metrics.mape:.2%}, "
            f"RMSE={overall_metrics.rmse:.2f}, R²={overall_metrics.r2:.3f}"
        )

        return overall_metrics, project_evaluations

    def _evaluate_project(
        self,
        model,
        dataset_service,
        project_id: int,
        mapped_project_id: int,
        trade_id_map: Dict,
        seq_len: int,
        start_date: date,
        end_date: date,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate predictions for a project in the test period."""
        predictions = []
        actuals = []
        stds = []

        # Get features for this project
        features = self.db.query(CanonicalCostFeature).filter(
            CanonicalCostFeature.project_id == project_id,
            CanonicalCostFeature.period_date >= start_date - timedelta(days=seq_len * 35),
            CanonicalCostFeature.period_date <= end_date,
        ).order_by(
            CanonicalCostFeature.canonical_trade_id,
            CanonicalCostFeature.period_date
        ).all()

        if not features:
            return [], [], []

        # Group by trade
        trade_features: Dict[int, List] = {}
        for f in features:
            if f.canonical_trade_id not in trade_features:
                trade_features[f.canonical_trade_id] = []
            trade_features[f.canonical_trade_id].append(f)

        for trade_id, trade_data in trade_features.items():
            mapped_trade_id = trade_id_map.get(str(trade_id), 0)

            # Find points in test period
            for i, f in enumerate(trade_data):
                if f.period_date < start_date or i < seq_len:
                    continue

                # Build sequence from preceding data
                seq_data = []
                for j in range(seq_len):
                    prev_f = trade_data[i - seq_len + j]
                    seq_data.append([
                        prev_f.cost_per_sf_cents or 0,
                        prev_f.cumulative_cost_per_sf_cents or 0,
                        prev_f.budget_per_sf_cents or 0,
                        prev_f.pct_complete or 0,
                        prev_f.schedule_pct_elapsed or 0,
                    ])

                # Normalize
                seq_array = np.array(seq_data, dtype=np.float32)
                for feat_idx, feat_name in enumerate(dataset_service.FEATURE_NAMES):
                    mean, std = dataset_service._feature_stats.get(feat_name, (0, 1))
                    seq_array[:, feat_idx] = (seq_array[:, feat_idx] - mean) / std

                # Predict
                try:
                    result = model.predict_with_uncertainty(
                        seq_array,
                        project_id=mapped_project_id,
                        trade_id=mapped_trade_id,
                    )
                    predictions.append(result.mean)
                    stds.append(result.std)
                    actuals.append(f.cumulative_cost_per_sf_cents or 0)
                except Exception as e:
                    logger.debug(f"Prediction failed: {e}")
                    continue

        return predictions, actuals, stds

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        stds: np.ndarray,
    ) -> EvaluationMetrics:
        """Compute evaluation metrics from predictions and actuals."""
        if len(predictions) == 0:
            return EvaluationMetrics()

        # Basic metrics
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2

        mae = float(np.mean(abs_errors))
        mse = float(np.mean(sq_errors))
        rmse = float(np.sqrt(mse))

        # MAPE (avoiding division by zero)
        non_zero_mask = np.abs(actuals) > 1e-6
        if np.any(non_zero_mask):
            mape = float(np.mean(abs_errors[non_zero_mask] / np.abs(actuals[non_zero_mask])))
        else:
            mape = 0.0

        # R-squared
        ss_res = np.sum(sq_errors)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Coverage metrics (for probabilistic models)
        from scipy import stats
        z_80 = stats.norm.ppf(0.9)
        z_90 = stats.norm.ppf(0.95)

        lower_80 = predictions - z_80 * stds
        upper_80 = predictions + z_80 * stds
        coverage_80 = float(np.mean((actuals >= lower_80) & (actuals <= upper_80)))

        lower_90 = predictions - z_90 * stds
        upper_90 = predictions + z_90 * stds
        coverage_90 = float(np.mean((actuals >= lower_90) & (actuals <= upper_90)))

        avg_interval_width = float(np.mean(2 * z_80 * stds))

        return EvaluationMetrics(
            mape=mape,
            mae=mae,
            rmse=rmse,
            mse=mse,
            r2=r2,
            coverage_80=coverage_80,
            coverage_90=coverage_90,
            avg_interval_width=avg_interval_width,
            num_samples=len(predictions),
        )

    def backtest(
        self,
        model_id: int,
        start_date: date,
        end_date: date,
        step_months: int = 1,
    ) -> BacktestResult:
        """
        Perform walk-forward backtesting.

        Simulates training on historical data and predicting forward,
        then stepping through time to assess model stability.

        Args:
            model_id: Model to backtest
            start_date: Start of backtest period
            end_date: End of backtest period
            step_months: Months between backtest points

        Returns:
            BacktestResult with per-period metrics
        """
        period_metrics = []
        all_project_evals = []

        current_date = start_date
        while current_date <= end_date:
            # Evaluate for this period
            period_end = current_date + timedelta(days=step_months * 30)
            metrics, project_evals = self.evaluate_model(
                model_id,
                test_start_date=current_date,
                test_end_date=period_end,
            )

            period_metrics.append((current_date, metrics))
            all_project_evals.extend(project_evals)

            current_date = period_end

        # Aggregate overall metrics
        if period_metrics:
            all_mape = [m.mape for _, m in period_metrics if m.num_samples > 0]
            all_rmse = [m.rmse for _, m in period_metrics if m.num_samples > 0]
            all_r2 = [m.r2 for _, m in period_metrics if m.num_samples > 0]

            overall = EvaluationMetrics(
                mape=float(np.mean(all_mape)) if all_mape else 0.0,
                rmse=float(np.mean(all_rmse)) if all_rmse else 0.0,
                r2=float(np.mean(all_r2)) if all_r2 else 0.0,
                num_samples=sum(m.num_samples for _, m in period_metrics),
            )
        else:
            overall = EvaluationMetrics()

        return BacktestResult(
            model_id=model_id,
            backtest_periods=len(period_metrics),
            overall_metrics=overall,
            project_evaluations=all_project_evals,
            period_metrics=period_metrics,
        )

    def compare_models(
        self,
        model_ids: List[int],
        test_start_date: Optional[date] = None,
        test_end_date: Optional[date] = None,
    ) -> Dict[int, EvaluationMetrics]:
        """
        Compare multiple models on the same test set.

        Args:
            model_ids: List of model IDs to compare
            test_start_date: Start of test period
            test_end_date: End of test period

        Returns:
            Dict mapping model_id to metrics
        """
        results = {}

        for model_id in model_ids:
            metrics, _ = self.evaluate_model(
                model_id, test_start_date, test_end_date
            )
            results[model_id] = metrics

        # Log comparison
        logger.info("Model comparison results:")
        for model_id, metrics in results.items():
            logger.info(
                f"  Model {model_id}: MAPE={metrics.mape:.2%}, "
                f"RMSE={metrics.rmse:.2f}, R²={metrics.r2:.3f}"
            )

        return results

    def get_worst_performing_projects(
        self,
        model_id: int,
        top_n: int = 5,
    ) -> List[ProjectEvaluation]:
        """
        Identify worst-performing projects for targeted improvement.

        Args:
            model_id: Model to evaluate
            top_n: Number of worst projects to return

        Returns:
            List of worst-performing project evaluations
        """
        _, project_evals = self.evaluate_model(model_id)

        # Sort by MAPE descending
        sorted_evals = sorted(
            project_evals,
            key=lambda x: x.metrics.mape,
            reverse=True
        )

        return sorted_evals[:top_n]

    def assess_calibration(
        self,
        model_id: int,
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
    ) -> Dict[float, float]:
        """
        Assess probabilistic calibration across confidence levels.

        A well-calibrated model should have actual coverage close to
        the stated confidence level.

        Args:
            model_id: Model to assess
            confidence_levels: List of confidence levels to check

        Returns:
            Dict mapping confidence_level to actual coverage
        """
        import json
        from pathlib import Path
        from scipy import stats
        from app.forecasting.models import get_multi_project_forecaster
        from .training_dataset_service import TrainingDatasetService

        # Load model and get predictions
        model_record = self.db.query(MLModelRegistry).get(model_id)
        if not model_record:
            return {}

        model_dir = Path(model_record.model_path)
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Collect predictions with uncertainty
        all_predictions = []
        all_stds = []
        all_actuals = []

        MultiProjectForecaster = get_multi_project_forecaster()
        model = MultiProjectForecaster(
            num_projects=len(metadata['id_mappings']['project_id_map']),
            num_trades=len(metadata['id_mappings']['trade_id_map']),
            **metadata['config'],
        )
        model.load(str(model_dir))

        dataset_service = TrainingDatasetService(self.db)
        dataset_service.set_feature_stats(metadata['feature_stats'])
        dataset_service._project_id_map = metadata['id_mappings']['project_id_map']
        dataset_service._trade_id_map = metadata['id_mappings']['trade_id_map']

        test_end = date.today()
        test_start = test_end - timedelta(days=90)

        for project_id, mapped_id in metadata['id_mappings']['project_id_map'].items():
            pred, actual, std = self._evaluate_project(
                model, dataset_service, int(project_id), mapped_id,
                metadata['id_mappings']['trade_id_map'],
                metadata['config']['seq_len'],
                test_start, test_end
            )
            all_predictions.extend(pred)
            all_actuals.extend(actual)
            all_stds.extend(std)

        if not all_predictions:
            return {}

        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        stds = np.array(all_stds)

        # Compute coverage at each level
        calibration = {}
        for level in confidence_levels:
            z = stats.norm.ppf((1 + level) / 2)
            lower = predictions - z * stds
            upper = predictions + z * stds
            coverage = float(np.mean((actuals >= lower) & (actuals <= upper)))
            calibration[level] = coverage

        return calibration

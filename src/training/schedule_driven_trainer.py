"""
Schedule-Driven Training Pipeline

Complete training pipeline with schedule as primary driver.

ORDER MATTERS:
1. Parse schedule FIRST
2. Allocate expected costs to activities
3. Map actual costs to trades
4. Build schedule-driven features
5. Train models
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import pickle
import logging

# ═══════════════════════════════════════════════════════════════════════════
# QUICK WIN #1: Enable Mixed Precision Training (GPU only)
# Provides 30-50% throughput improvement on compatible GPUs
# ═══════════════════════════════════════════════════════════════════════════
try:
    # Only enable mixed precision if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
except Exception:
    pass  # Fall back to default precision if not supported


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Sequence Length Configuration
# Allows training trades with less historical data
# Thresholds ensure adequate training samples: data - SL >= min_sequences (6)
# ═══════════════════════════════════════════════════════════════════════════
SEQUENCE_CONFIGS = [
    {'min_months': 24, 'sequence_length': 12, 'tier': 'full_seasonal'},  # 24-12=12 samples
    {'min_months': 12, 'sequence_length': 6,  'tier': 'quarterly'},      # 12-6=6 samples
    {'min_months': 5,  'sequence_length': 3,  'tier': 'minimal'},        # 5-3=2 samples
]


def assign_sequence_tier(history_length: int) -> Optional[dict]:
    """Assign the longest viable sequence length for each trade."""
    for config in SEQUENCE_CONFIGS:
        if history_length >= config['min_months']:
            return config
    return None  # Exclude if < 5 months


from ..schedule.parser import ScheduleParser
from ..schedule.cost_allocator import ActivityCostAllocator
from ..features.schedule_driven_features import ScheduleDrivenFeatureBuilder
from ..models.schedule_driven_model import ScheduleDrivenModel, create_schedule_driven_model

logger = logging.getLogger(__name__)


@dataclass
class TradeModelResult:
    """Results from training a single trade model"""
    trade: str
    final_loss: float
    samples: int
    epochs_trained: int
    best_epoch: int
    sequence_length: int = 6
    max_gradient_norm: float = 0.0


@dataclass
class ForecastResult:
    """Forecast result for a trade"""
    trade: str
    current_phase: str
    project_pct_complete: float
    trade_phase_active: bool
    gmp_budget: float
    spent_to_date: float
    expected_by_schedule: float
    schedule_variance: float
    variance_pct: float
    forecast_next_month: float
    forecast_std: float
    forecast_at_completion: float
    budget_variance: float


class ScheduleDrivenTrainer:
    """
    Complete training pipeline with schedule as primary driver.

    The key insight: Schedule position predicts cost better than time alone.
    """

    def __init__(
        self,
        sequence_length: int = 6,
        schedule_features: int = 12,
        trade_features: int = 8,
        cost_features: int = 4
    ):
        """
        Initialize trainer.

        Args:
            sequence_length: Number of periods in input sequence
            schedule_features: Number of schedule features
            trade_features: Number of trade context features
            cost_features: Number of cost features
        """
        self.sequence_length = sequence_length
        self.schedule_features = schedule_features
        self.trade_features = trade_features
        self.cost_features = cost_features

        self.parser: Optional[ScheduleParser] = None
        self.allocator: Optional[ActivityCostAllocator] = None
        self.feature_builder: Optional[ScheduleDrivenFeatureBuilder] = None
        self.models: Dict[str, ScheduleDrivenModel] = {}
        self.trade_data: Dict[str, pd.DataFrame] = {}
        self.scalers: Dict[str, dict] = {}
        # Track per-trade sequence lengths for dynamic sizing
        self.trade_sequence_lengths: Dict[str, int] = {}

    def prepare(
        self,
        schedule_df: pd.DataFrame,
        gmp_breakdown_df: pd.DataFrame,
        direct_costs_df: pd.DataFrame,
        budget_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Prepare training data with schedule as primary.

        ORDER MATTERS:
        1. Parse schedule FIRST
        2. Allocate expected costs to activities
        3. Map actual costs to trades
        4. Build schedule-driven features
        """
        logger.info("=" * 60)
        logger.info("SCHEDULE-DRIVEN TRAINING PIPELINE")
        logger.info("=" * 60)

        # Reset index on input DataFrames to prevent "Unalignable boolean Series" errors
        # This ensures clean boolean indexing operations throughout the pipeline
        direct_costs_df = direct_costs_df.copy().reset_index(drop=True)
        schedule_df = schedule_df.copy().reset_index(drop=True)
        gmp_breakdown_df = gmp_breakdown_df.copy().reset_index(drop=True)

        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Parse schedule (PRIMARY)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n[1/4] Parsing schedule (PRIMARY)...")
        self.parser = ScheduleParser(schedule_df)

        logger.info(f"      Project: {self.parser.project_start} to {self.parser.project_end}")
        logger.info(f"      Activities: {len(self.parser.activities)}")
        logger.info(f"      Phases: {len(self.parser.phases)}")

        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Allocate expected costs to activities
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n[2/4] Allocating expected costs to activities...")
        self.allocator = ActivityCostAllocator(self.parser, gmp_breakdown_df)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Map actual costs to trades (using activity mappings)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n[3/4] Mapping actual costs to GMP trades...")

        # Use the schedule parser's trade mapping
        def map_cost_to_trade(row):
            """Map a direct cost to a trade based on cost code or name"""
            cost_code = str(row.get('cost_code', '') or row.get('Cost Code', ''))
            name = str(row.get('name', '') or row.get('Description', ''))

            # Try to find matching activity
            primary, secondary, weight, _ = self.parser._map_to_trade(name)
            return primary

        direct_costs_df['gmp_trade'] = direct_costs_df.apply(map_cost_to_trade, axis=1)

        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Build schedule-driven features
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n[4/4] Building schedule-driven features...")
        self.feature_builder = ScheduleDrivenFeatureBuilder(
            self.parser,
            self.allocator,
            gmp_breakdown_df
        )

        # Ensure we have date and amount columns
        date_col = next((c for c in ['date', 'Date', 'transaction_date'] if c in direct_costs_df.columns), None)
        amount_col = next((c for c in ['amount', 'Amount', 'total'] if c in direct_costs_df.columns), None)

        if date_col and amount_col:
            direct_costs_df['_date'] = pd.to_datetime(direct_costs_df[date_col], errors='coerce')
            direct_costs_df['year_month'] = direct_costs_df['_date'].dt.to_period('M').astype(str)

            # Build training data for each trade with dynamic sequence lengths
            excluded_trades = []
            for trade in direct_costs_df['gmp_trade'].dropna().unique():
                trade_costs = direct_costs_df[direct_costs_df['gmp_trade'] == trade]

                # Aggregate to monthly
                monthly = trade_costs.groupby('year_month').agg({
                    amount_col: 'sum'
                }).reset_index()
                monthly.columns = ['year_month', 'total_cost']
                monthly = monthly.sort_values('year_month')

                # Dynamic sequence length assignment (key improvement)
                tier_config = assign_sequence_tier(len(monthly))
                if tier_config is None:
                    excluded_trades.append((trade, len(monthly)))
                    logger.debug(f"Skipping {trade}: only {len(monthly)} months (need >= 5)")
                    continue

                # Store the sequence length for this trade
                self.trade_sequence_lengths[trade] = tier_config['sequence_length']
                logger.debug(
                    f"{trade}: {len(monthly)} months -> "
                    f"SL={tier_config['sequence_length']} ({tier_config['tier']})"
                )

                # Build schedule-driven features
                trade_df = self.feature_builder.build_training_data(trade, monthly)
                self.trade_data[trade] = trade_df

        # Log summary of trade eligibility
        logger.info(f"\n      Built training data for {len(self.trade_data)} trades")
        if excluded_trades:
            logger.info(f"      Excluded {len(excluded_trades)} trades with < 5 months data")
            for trade, months in excluded_trades[:5]:  # Show first 5
                logger.debug(f"        - {trade}: {months} months")

    def _create_sequences(
        self,
        trade_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training sequences.

        Returns:
            (schedule_seq, trade_context, cost_seq, targets)
        """
        df = self.trade_data[trade_name].copy()

        # Schedule sequence features (Level 1)
        schedule_cols = [
            'project_pct_complete', 'project_days_remaining',
            'phase_pct_complete', 'phase_days_remaining',
            'total_active_activities', 'critical_path_activities',
            'activities_completed_to_date', 'activities_remaining',
            'avg_float', 'min_float',
            'trade_expected_pct_complete', 'phases_active_count'
        ]

        # Trade context features (Level 2 - static per period)
        trade_cols = [
            'trade_activities_active', 'trade_activities_complete',
            'trade_activities_remaining', 'trade_phase_active',
            'trade_expected_pct_complete', 'trade_gmp_budget',
            'trade_budget_remaining', 'trade_actual_pct_spent'
        ]

        # Cost sequence features (Level 3)
        cost_cols = [
            'monthly_cost', 'cumulative_cost',
            'trade_schedule_variance', 'trade_variance_pct'
        ]

        # Ensure columns exist
        for col in schedule_cols + trade_cols + cost_cols:
            if col not in df.columns:
                df[col] = 0

        # ═══════════════════════════════════════════════════════════════════
        # Improved Normalization Strategy
        # - Log-transform monetary columns to handle large value ranges
        # - Robust z-score normalization with clipping
        # ═══════════════════════════════════════════════════════════════════

        # Columns requiring log-transform (monetary values with large ranges)
        log_transform_cols = {
            'monthly_cost', 'cumulative_cost', 'trade_gmp_budget',
            'trade_budget_remaining', 'trade_schedule_variance'
        }

        scaler_stats = {}
        for col in schedule_cols + trade_cols + cost_cols:
            if col not in df.columns:
                continue

            values = df[col].astype(float).copy()  # Ensure float type

            # Apply log1p transform for monetary columns (handles zeros)
            if col in log_transform_cols:
                # Handle negative values (e.g., negative variance)
                sign = np.sign(values)
                values = sign * np.log1p(np.abs(values))
                scaler_stats[col] = {'transform': 'log1p'}
            else:
                scaler_stats[col] = {'transform': 'none'}

            # Robust z-score: use median and IQR for outlier resistance
            median = values.median()
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = q75 - q25

            if iqr > 1e-6:
                # Scale by IQR (more robust than std)
                values = (values - median) / iqr
            elif values.std() > 1e-6:
                # Fall back to standard z-score
                values = (values - values.mean()) / values.std()

            # Clip extreme values to [-5, 5] range
            values = values.clip(-5, 5)

            df[col] = values
            scaler_stats[col].update({
                'median': float(median),
                'iqr': float(iqr) if iqr > 1e-6 else 1.0
            })

        self.scalers[trade_name] = scaler_stats

        schedule_values = df[schedule_cols].fillna(0).values
        trade_values = df[trade_cols].fillna(0).values
        cost_values = df[cost_cols].fillna(0).values
        targets = df['monthly_cost'].values

        X_schedule = []
        X_trade = []
        X_cost = []
        y = []

        # Use trade-specific sequence length if available
        seq_len = self.trade_sequence_lengths.get(trade_name, self.sequence_length)

        for i in range(seq_len, len(df)):
            X_schedule.append(schedule_values[i - seq_len:i])
            X_trade.append(trade_values[i])
            X_cost.append(cost_values[i - seq_len:i])
            y.append(targets[i])

        return (
            np.array(X_schedule, dtype=np.float32),
            np.array(X_trade, dtype=np.float32),
            np.array(X_cost, dtype=np.float32),
            np.array(y, dtype=np.float32)[:, np.newaxis]
        )

    def train(
        self,
        epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 15,
        min_samples: int = 4  # Lowered from 8 to train more trades
    ) -> Dict[str, TradeModelResult]:
        """
        Train models for all trades.

        Args:
            epochs: Maximum training epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience
            min_samples: Minimum samples required

        Returns:
            Dict of trade_name -> TradeModelResult
        """
        results = {}

        for trade_name, df in self.trade_data.items():
            if len(df) < min_samples:
                logger.info(f"Skipping {trade_name}: insufficient data ({len(df)} rows)")
                continue

            X_sched, X_trade, X_cost, y = self._create_sequences(trade_name)

            if len(X_sched) < 2:
                logger.info(f"Skipping {trade_name}: not enough sequences ({len(X_sched)})")
                continue

            # Build model
            model = create_schedule_driven_model(
                schedule_features=X_sched.shape[2],
                trade_features=X_trade.shape[1],
                cost_features=X_cost.shape[2],
                sequence_length=X_sched.shape[1]
            )

            # Training with XLA-compiled step and gradient tracking
            optimizer = keras.optimizers.Adam(learning_rate)

            best_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            max_grad_norm = 0.0
            inputs = [X_sched, X_trade, X_cost]

            for epoch in range(epochs):
                # Training step with gradient clipping
                with tf.GradientTape() as tape:
                    mean, std = model(inputs, training=True)

                    # Gaussian NLL loss
                    base_loss = tf.reduce_mean(
                        0.5 * tf.math.log(2 * np.pi * tf.square(std) + 1e-6) +
                        tf.square(y - mean) / (2 * tf.square(std) + 1e-6)
                    )

                    # Weight by trade phase active
                    phase_weight = tf.reduce_mean(X_trade[:, 3])
                    loss = base_loss * (1.0 + 0.3 * phase_weight)

                # Compute and clip gradients
                gradients = tape.gradient(loss, model.trainable_variables)
                gradients, grad_norm = tf.clip_by_global_norm(gradients, clip_norm=1.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Track gradient statistics for monitoring
                grad_norm_val = float(grad_norm.numpy())
                max_grad_norm = max(max_grad_norm, grad_norm_val)

                # Alert on gradient issues (threshold adjusted for clipped norms)
                if grad_norm_val > 100.0:
                    logger.warning(
                        f"{trade_name} epoch {epoch}: high gradient norm {grad_norm_val:.2f}"
                    )

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            model._is_trained = True
            self.models[trade_name] = model

            seq_len = self.trade_sequence_lengths.get(trade_name, self.sequence_length)
            result = TradeModelResult(
                trade=trade_name,
                final_loss=float(loss.numpy()),
                samples=len(X_sched),
                epochs_trained=epoch + 1,
                best_epoch=best_epoch,
                sequence_length=seq_len,
                max_gradient_norm=max_grad_norm
            )
            results[trade_name] = result

            logger.info(
                f"Trained {trade_name}: loss={loss.numpy():.4f}, "
                f"samples={len(X_sched)}, epochs={epoch + 1}, "
                f"SL={seq_len}, max_grad={max_grad_norm:.2f}"
            )

        return results

    def forecast(self, trade_name: str) -> Optional[ForecastResult]:
        """
        Generate forecast for a trade.

        Args:
            trade_name: Name of trade to forecast

        Returns:
            ForecastResult or None if model not available
        """
        if trade_name not in self.models:
            return None

        model = self.models[trade_name]
        df = self.trade_data[trade_name]

        X_sched, X_trade, X_cost, _ = self._create_sequences(trade_name)

        if len(X_sched) == 0:
            return None

        # Use last sequence
        mean, std = model(
            [X_sched[-1:], X_trade[-1:], X_cost[-1:]],
            training=False
        )

        last_row = df.iloc[-1]

        # Denormalize prediction using scaler stats
        scaler = self.scalers.get(trade_name, {})
        mean_val = float(mean.numpy()[0, 0])
        std_val = float(std.numpy()[0, 0])

        if 'monthly_cost' in scaler:
            stats = scaler['monthly_cost']
            # Reverse robust z-score normalization
            iqr = stats.get('iqr', 1.0)
            median = stats.get('median', 0.0)
            mean_val = mean_val * iqr + median
            std_val = std_val * iqr

            # Reverse log1p transform if applied
            if stats.get('transform') == 'log1p':
                # For log1p: x = sign(y) * (exp(|y|) - 1)
                sign = 1 if mean_val >= 0 else -1
                mean_val = sign * (np.expm1(abs(mean_val)))
                std_val = abs(np.expm1(abs(std_val)))

        # Calculate forecast at completion
        budget = last_row.get('trade_gmp_budget', 0)
        spent = last_row.get('cumulative_cost', 0)
        expected = last_row.get('trade_expected_cost', 0)

        if expected > 0:
            cpi = expected / spent if spent > 0 else 1.0
            remaining_expected = budget - expected
            forecast_remaining = remaining_expected / max(0.5, min(2.0, cpi))
            forecast_at_completion = spent + forecast_remaining
        else:
            forecast_at_completion = spent + mean_val * 6  # Rough estimate

        variance_pct = (spent - expected) / expected if expected > 0 else 0

        return ForecastResult(
            trade=trade_name,
            current_phase=str(last_row.get('current_phase', 'UNKNOWN')),
            project_pct_complete=float(last_row.get('project_pct_complete', 0)),
            trade_phase_active=bool(last_row.get('trade_phase_active', False)),
            gmp_budget=float(budget),
            spent_to_date=float(spent),
            expected_by_schedule=float(expected),
            schedule_variance=float(spent - expected),
            variance_pct=float(variance_pct),
            forecast_next_month=mean_val,
            forecast_std=std_val,
            forecast_at_completion=float(forecast_at_completion),
            budget_variance=float(forecast_at_completion - budget)
        )

    def save(self, path: str) -> None:
        """Save trained models and state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models
        for trade_name, model in self.models.items():
            safe_name = trade_name.replace(' ', '_').replace('&', 'and')
            model_path = path / f"{safe_name}_model.weights.h5"
            model.save_weights(str(model_path))

        # Save scalers and config
        config = {
            'scalers': self.scalers,
            'trades': list(self.models.keys()),
            'sequence_length': self.sequence_length,
            'trade_sequence_lengths': self.trade_sequence_lengths
        }
        with open(path / 'config.pkl', 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"Saved {len(self.models)} models to {path}")

    def load(self, path: str) -> None:
        """Load trained models"""
        path = Path(path)

        # Load config
        with open(path / 'config.pkl', 'rb') as f:
            config = pickle.load(f)

        self.scalers = config['scalers']
        self.sequence_length = config['sequence_length']
        # Restore trade-specific sequence lengths (backward compatible)
        self.trade_sequence_lengths = config.get('trade_sequence_lengths', {})

        # Load models
        for trade_name in config['trades']:
            safe_name = trade_name.replace(' ', '_').replace('&', 'and')
            model_path = path / f"{safe_name}_model.weights.h5"

            if model_path.exists():
                # Use trade-specific sequence length
                seq_len = self.trade_sequence_lengths.get(
                    trade_name, self.sequence_length
                )
                model = create_schedule_driven_model(
                    sequence_length=seq_len
                )
                model.load_weights(str(model_path))
                model._is_trained = True
                self.models[trade_name] = model

        logger.info(f"Loaded {len(self.models)} models from {path}")

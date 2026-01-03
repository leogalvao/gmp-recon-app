"""
ML Forecasting Module for GMP Reconciliation App.
Predicts remaining cost by region and GMP division.
Uses LinearRegression baseline with optional PyTorch MLP.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - using LinearRegression only")


MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
MIN_DATA_POINTS_MLP = 18  # Minimum data points to train MLP


# Conditional PyTorch MLP class definition
if TORCH_AVAILABLE:
    class SimpleMLP(nn.Module):
        """Simple 2-layer MLP for remaining cost prediction."""
        
        def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1)
            )
        
        def forward(self, x):
            return self.net(x)
else:
    # Placeholder when PyTorch not available
    SimpleMLP = None


def create_time_features(dates: pd.Series) -> pd.DataFrame:
    """
    Create time-based features from dates.
    Features: month, quarter, day_of_year, week_of_year
    """
    df = pd.DataFrame()
    df['month'] = dates.dt.month
    df['quarter'] = dates.dt.quarter
    df['day_of_year'] = dates.dt.dayofyear
    df['week_of_year'] = dates.dt.isocalendar().week.astype(int)
    df['year'] = dates.dt.year
    return df


def create_rolling_features(amounts: pd.Series, dates: pd.Series, 
                            windows: List[int] = [30, 60, 90]) -> pd.DataFrame:
    """
    Create rolling sum features for different time windows.
    """
    df = pd.DataFrame({'date': dates, 'amount': amounts})
    df = df.sort_values('date')
    
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        # Calculate rolling sum with date-aware windowing
        result[f'rolling_{window}d'] = (
            df.set_index('date')['amount']
            .rolling(f'{window}D', min_periods=1)
            .sum()
            .values
        )
    
    return result


def create_lag_features(df: pd.DataFrame, group_col: str, 
                        value_col: str, lags: List[int] = [1, 2]) -> pd.DataFrame:
    """
    Create lagged features for time series forecasting.
    Lags are in months.
    """
    result = df.copy()
    
    # Create month period for grouping
    result['month_period'] = result['date_parsed'].dt.to_period('M')
    
    for lag in lags:
        lag_col = f'lag_{lag}m'
        result[lag_col] = (
            result.groupby(group_col)[value_col]
            .shift(lag)
        )
    
    return result


def prepare_features(
    direct_costs_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    gmp_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare feature matrix for ML training.
    
    Features:
    - Time features: month, quarter, day_of_year
    - Rolling sums: 30/60/90-day windows
    - Lagged actuals: t-1, t-2 months
    - Budget ratio: current_budget / gmp_amount_total
    
    Returns:
    - Feature DataFrame with target column
    - Feature metadata dictionary
    """
    # Filter to as_of_date
    if as_of_date:
        df = direct_costs_df[direct_costs_df['date_parsed'] <= as_of_date].copy()
    else:
        df = direct_costs_df.copy()
    
    # Exclude flagged duplicates
    df = df[df['excluded_from_actuals'] == False]
    
    if len(df) == 0:
        return pd.DataFrame(), {}
    
    # Add time features
    time_feats = create_time_features(df['date_parsed'])
    df = pd.concat([df, time_feats], axis=1)
    
    # Aggregate by GMP division and region (weekly)
    df['week'] = df['date_parsed'].dt.to_period('W')
    
    # Join to budget to get GMP division
    budget_gmp = budget_df[['Budget Code', 'gmp_division']].drop_duplicates()
    df = df.merge(budget_gmp, left_on='mapped_budget_code', right_on='Budget Code', how='left')
    
    # Create weekly aggregation
    weekly_agg = df.groupby(['gmp_division', 'week']).agg({
        'amount_west': 'sum',
        'amount_east': 'sum',
        'amount_cents': 'sum',
        'month': 'first',
        'quarter': 'first',
        'day_of_year': 'mean',
        'week_of_year': 'first',
        'year': 'first'
    }).reset_index()
    
    # Add GMP amounts for ratio feature
    gmp_amounts = gmp_df[['GMP', 'amount_total_cents']].copy()
    gmp_amounts.columns = ['gmp_division', 'gmp_amount_cents']
    weekly_agg = weekly_agg.merge(gmp_amounts, on='gmp_division', how='left')
    
    # Calculate budget ratio
    weekly_agg['budget_ratio'] = weekly_agg.apply(
        lambda r: r['amount_cents'] / r['gmp_amount_cents'] if r['gmp_amount_cents'] > 0 else 0,
        axis=1
    )
    
    # Add cumulative sum per division
    weekly_agg = weekly_agg.sort_values(['gmp_division', 'week'])
    weekly_agg['cumsum_west'] = weekly_agg.groupby('gmp_division')['amount_west'].cumsum()
    weekly_agg['cumsum_east'] = weekly_agg.groupby('gmp_division')['amount_east'].cumsum()
    weekly_agg['cumsum_total'] = weekly_agg.groupby('gmp_division')['amount_cents'].cumsum()
    
    # Calculate remaining as target (GMP - cumsum at each point)
    weekly_agg['remaining_west'] = weekly_agg['gmp_amount_cents'] / 2 - weekly_agg['cumsum_west']
    weekly_agg['remaining_east'] = weekly_agg['gmp_amount_cents'] / 2 - weekly_agg['cumsum_east']
    
    feature_cols = ['month', 'quarter', 'day_of_year', 'week_of_year', 'year',
                    'budget_ratio', 'cumsum_west', 'cumsum_east', 'cumsum_total']
    
    metadata = {
        'feature_cols': feature_cols,
        'target_cols': ['remaining_west', 'remaining_east'],
        'n_samples': len(weekly_agg),
        'n_divisions': weekly_agg['gmp_division'].nunique()
    }
    
    return weekly_agg, metadata


def train_linear_model(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, float]:
    """
    Train LinearRegression model with time series cross-validation.
    Returns model and mean MAE from CV.
    """
    if len(X) < 5:
        # Not enough data for CV, just fit
        model = LinearRegression()
        model.fit(X, y)
        return model, 0.0
    
    n_splits = min(5, len(X) // 3)
    if n_splits < 2:
        model = LinearRegression()
        model.fit(X, y)
        return model, 0.0
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
    
    # Final fit on all data
    model = LinearRegression()
    model.fit(X, y)
    
    return model, np.mean(maes)


def train_mlp_model(X: np.ndarray, y: np.ndarray, 
                    epochs: int = 100, patience: int = 10) -> Tuple[Optional[SimpleMLP], float]:
    """
    Train PyTorch MLP model with early stopping.
    Returns model and mean MAE from validation.
    """
    if not TORCH_AVAILABLE or len(X) < MIN_DATA_POINTS_MLP:
        return None, float('inf')
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    if len(X_val) < 3:
        return None, float('inf')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train.reshape(-1, 1))
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val.reshape(-1, 1))
    
    # Initialize model
    model = SimpleMLP(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Calculate MAE on validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).numpy().flatten()
        mae = mean_absolute_error(y_val, val_preds)
    
    return model, mae


class ForecastingPipeline:
    """
    Main forecasting pipeline that trains models per GMP division × region.
    """
    
    def __init__(self):
        self.models = {}  # {(gmp_division, region): {'model': ..., 'scaler': ..., 'type': ...}}
        self.last_trained = None
        self.training_stats = {}
    
    def train(self, direct_costs_df: pd.DataFrame, budget_df: pd.DataFrame,
              gmp_df: pd.DataFrame, as_of_date: Optional[datetime] = None):
        """
        Train forecasting models for all GMP divisions.
        """
        features_df, metadata = prepare_features(
            direct_costs_df, budget_df, gmp_df, as_of_date
        )
        
        if len(features_df) == 0:
            print("No data available for training")
            return
        
        feature_cols = metadata['feature_cols']
        self.training_stats = {
            'n_samples': metadata['n_samples'],
            'n_divisions': metadata['n_divisions'],
            'models_trained': 0,
            'divisions': {}
        }
        
        for division in features_df['gmp_division'].unique():
            div_data = features_df[features_df['gmp_division'] == division]
            
            if len(div_data) < 3:
                continue
            
            X = div_data[feature_cols].values
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            for region in ['west', 'east']:
                y = div_data[f'remaining_{region}'].values
                y = np.nan_to_num(y, nan=0.0)
                
                # Train linear model
                lr_model, lr_mae = train_linear_model(X, y)
                
                # Try MLP if enough data
                mlp_model, mlp_mae = train_mlp_model(X, y)
                
                # Choose best model
                if mlp_model is not None and mlp_mae < lr_mae:
                    self.models[(division, region)] = {
                        'model': mlp_model,
                        'scaler': StandardScaler().fit(X),
                        'type': 'mlp',
                        'mae': mlp_mae
                    }
                else:
                    self.models[(division, region)] = {
                        'model': lr_model,
                        'scaler': None,
                        'type': 'linear',
                        'mae': lr_mae
                    }
                
                self.training_stats['models_trained'] += 1
            
            self.training_stats['divisions'][division] = {
                'n_samples': len(div_data),
                'model_type_west': self.models.get((division, 'west'), {}).get('type', 'none'),
                'model_type_east': self.models.get((division, 'east'), {}).get('type', 'none')
            }
        
        self.last_trained = datetime.utcnow()
    
    def predict(self, direct_costs_df: pd.DataFrame, budget_df: pd.DataFrame,
                gmp_df: pd.DataFrame, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate predictions for remaining cost by GMP division × region.
        
        Returns DataFrame with columns:
        - gmp_division
        - predicted_remaining_west
        - predicted_remaining_east
        """
        features_df, metadata = prepare_features(
            direct_costs_df, budget_df, gmp_df, as_of_date
        )
        
        if len(features_df) == 0:
            return pd.DataFrame(columns=['gmp_division', 'predicted_remaining_west', 'predicted_remaining_east'])
        
        feature_cols = metadata['feature_cols']
        results = []
        
        for division in gmp_df['GMP'].unique():
            # Get latest data point for this division
            div_data = features_df[features_df['gmp_division'] == division]
            
            pred_west = 0
            pred_east = 0
            
            if len(div_data) > 0:
                latest = div_data.iloc[-1]
                X = latest[feature_cols].values.reshape(1, -1)
                X = np.nan_to_num(X, nan=0.0)
                
                # Predict West
                if (division, 'west') in self.models:
                    model_info = self.models[(division, 'west')]
                    if model_info['scaler']:
                        X_scaled = model_info['scaler'].transform(X)
                    else:
                        X_scaled = X
                    
                    if model_info['type'] == 'mlp' and TORCH_AVAILABLE:
                        model_info['model'].eval()
                        with torch.no_grad():
                            pred_west = model_info['model'](torch.FloatTensor(X_scaled)).item()
                    else:
                        pred_west = model_info['model'].predict(X_scaled)[0]
                
                # Predict East
                if (division, 'east') in self.models:
                    model_info = self.models[(division, 'east')]
                    if model_info['scaler']:
                        X_scaled = model_info['scaler'].transform(X)
                    else:
                        X_scaled = X
                    
                    if model_info['type'] == 'mlp' and TORCH_AVAILABLE:
                        model_info['model'].eval()
                        with torch.no_grad():
                            pred_east = model_info['model'](torch.FloatTensor(X_scaled)).item()
                    else:
                        pred_east = model_info['model'].predict(X_scaled)[0]
            
            # Ensure non-negative
            pred_west = max(0, int(round(pred_west)))
            pred_east = max(0, int(round(pred_east)))
            
            results.append({
                'gmp_division': division,
                'predicted_remaining_west': pred_west,
                'predicted_remaining_east': pred_east
            })
        
        return pd.DataFrame(results)
    
    def get_training_status(self) -> Dict:
        """Get current training status."""
        return {
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'models_count': len(self.models),
            'stats': self.training_stats
        }


# Module-level pipeline instance
_pipeline = None


def get_forecasting_pipeline() -> ForecastingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ForecastingPipeline()
    return _pipeline

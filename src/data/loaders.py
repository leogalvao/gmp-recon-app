"""
Data Loaders for GMP Forecasting System

Loads all required data files:
- schedule.csv: Project schedule with activities
- direct_costs.csv: Actual cost transactions
- budget.csv: Budget codes and mappings
- breakdown.csv: GMP breakdown by trade
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for GMP forecasting.

    Handles various file formats and column naming conventions.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data files.

        Returns:
            Dict with keys: schedule, direct_costs, budget, breakdown
        """
        data = {}

        # Schedule
        schedule_files = ['schedule.csv', 'Schedule.csv', 'project_schedule.csv']
        data['schedule'] = self._load_first_found(schedule_files)

        # Direct costs
        cost_files = ['direct_costs.csv', 'Direct Costs.csv', 'costs.csv', 'DirectCosts.csv']
        data['direct_costs'] = self._load_first_found(cost_files)

        # Budget
        budget_files = ['budget.csv', 'Budget.csv', 'budget_codes.csv']
        data['budget'] = self._load_first_found(budget_files)

        # GMP breakdown
        breakdown_files = ['breakdown.csv', 'Breakdown.csv', 'gmp_breakdown.csv', 'GMP.csv']
        data['breakdown'] = self._load_first_found(breakdown_files)

        # Log what was loaded
        for name, df in data.items():
            if df is not None:
                logger.info(f"Loaded {name}: {len(df)} rows")
            else:
                logger.warning(f"Could not find {name} data file")

        return data

    def _load_first_found(self, filenames: list) -> Optional[pd.DataFrame]:
        """Try to load the first file that exists"""
        for filename in filenames:
            path = self.data_dir / filename
            if path.exists():
                try:
                    return pd.read_csv(path)
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
                    continue

        return None

    def load_schedule(self) -> Optional[pd.DataFrame]:
        """Load schedule file specifically"""
        files = ['schedule.csv', 'Schedule.csv', 'project_schedule.csv']
        return self._load_first_found(files)

    def load_direct_costs(self) -> Optional[pd.DataFrame]:
        """Load direct costs file"""
        files = ['direct_costs.csv', 'Direct Costs.csv', 'costs.csv']
        return self._load_first_found(files)

    def load_breakdown(self) -> Optional[pd.DataFrame]:
        """Load GMP breakdown file"""
        files = ['breakdown.csv', 'Breakdown.csv', 'gmp_breakdown.csv']
        return self._load_first_found(files)

    def load_budget(self) -> Optional[pd.DataFrame]:
        """Load budget codes file"""
        files = ['budget.csv', 'Budget.csv', 'budget_codes.csv']
        return self._load_first_found(files)


def prepare_monthly_costs(
    direct_costs: pd.DataFrame,
    trade_column: str = 'gmp_trade_name',
    amount_column: str = 'amount',
    date_column: str = 'date'
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate direct costs to monthly by trade.

    Args:
        direct_costs: DataFrame with cost transactions
        trade_column: Column with trade assignment
        amount_column: Column with cost amount
        date_column: Column with transaction date

    Returns:
        Dict of trade_name -> monthly cost DataFrame
    """
    # Make a copy and reset index to prevent "Unalignable boolean Series" errors
    # when input DataFrame has a non-contiguous index from prior filtering
    direct_costs = direct_costs.copy().reset_index(drop=True)

    # Ensure date is datetime
    direct_costs[date_column] = pd.to_datetime(direct_costs[date_column], errors='coerce')
    direct_costs['year_month'] = direct_costs[date_column].dt.to_period('M').astype(str)

    result = {}

    for trade in direct_costs[trade_column].dropna().unique():
        # Use .values for boolean mask to avoid index alignment issues
        mask = (direct_costs[trade_column] == trade).values
        trade_costs = direct_costs[mask]

        monthly = trade_costs.groupby('year_month').agg({
            amount_column: 'sum'
        }).reset_index()

        monthly.columns = ['year_month', 'total_cost']
        monthly = monthly.sort_values('year_month').reset_index(drop=True)

        if len(monthly) > 0:
            result[trade] = monthly

    return result

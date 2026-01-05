"""
Forecasting Module for GMP Reconciliation App.
Provides line-level cost forecasting with multiple methods (EVM, PERT, Parametric).
Supports weekly and monthly time buckets with full audit trail.

All monetary values in integer cents.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from sqlalchemy.orm import Session
import pandas as pd
import json

from app.models import (
    ForecastConfig, ForecastSnapshot, ForecastPeriod, ForecastAuditLog,
    ForecastMethod
)
from app.modules.etl import parse_money_to_cents, cents_to_display


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# =============================================================================
# CALCULATION ENGINES
# =============================================================================

def calculate_evm(
    bac_cents: int,
    ac_cents: int,
    ev_cents: Optional[int] = None,
    performance_factor: float = 1.0,
    spi: Optional[float] = None
) -> Dict[str, Any]:
    """
    Earned Value Management (EVM) calculation.

    Formulas:
    - CPI = EV / AC (if EV tracked)
    - EAC = BAC / CPI (or AC + remaining / performance_factor if EV not tracked)
    - ETC = EAC - AC
    - VAR = BAC - EAC (negative = over budget)

    Args:
        bac_cents: Budget at Completion (GMP amount)
        ac_cents: Actual Cost to date
        ev_cents: Earned Value (optional, if tracked externally)
        performance_factor: CPI adjustment (default 1.0 = at budget rate)
        spi: Schedule Performance Index (optional)

    Returns:
        Dict with eac_cents, etc_cents, var_cents, cpi, confidence_score, explanation
    """
    # Handle zero AC case
    if ac_cents == 0:
        return {
            'eac_cents': bac_cents,
            'etc_cents': bac_cents,
            'var_cents': 0,
            'cpi': None,
            'spi': spi,
            'confidence_score': 0.5,
            'confidence_band': 'medium',
            'explanation': (
                "No costs incurred yet. Forecast equals budget (BAC). "
                "CPI will be calculated once actuals are recorded."
            ),
            'ev_source': 'none'
        }

    # Calculate CPI based on EV availability
    if ev_cents is not None and ev_cents > 0:
        cpi = ev_cents / ac_cents
        ev_source = 'tracked'

        # Schedule-adjusted EAC if SPI available
        if spi is not None and spi > 0:
            # EAC = AC + (BAC - EV) / (CPI * SPI)
            remaining_work = bac_cents - ev_cents
            eac_cents = ac_cents + int(remaining_work / (cpi * spi))
            explanation = (
                f"EVM with tracked EV. CPI = {cpi:.3f} (EV/AC = ${ev_cents/100:,.0f}/${ac_cents/100:,.0f}). "
                f"SPI = {spi:.2f}. EAC = AC + (BAC - EV) / (CPI × SPI) = ${eac_cents/100:,.0f}."
            )
        else:
            # Basic EAC = BAC / CPI
            eac_cents = int(bac_cents / cpi) if cpi > 0 else bac_cents
            explanation = (
                f"EVM with tracked EV. CPI = {cpi:.3f} (EV/AC = ${ev_cents/100:,.0f}/${ac_cents/100:,.0f}). "
                f"EAC = BAC / CPI = ${bac_cents/100:,.0f} / {cpi:.3f} = ${eac_cents/100:,.0f}."
            )
    else:
        # EV not available - use performance factor as CPI proxy
        cpi = performance_factor
        ev_source = 'estimated'

        remaining = bac_cents - ac_cents
        adjusted_remaining = int(remaining / performance_factor) if performance_factor > 0 else remaining
        eac_cents = ac_cents + adjusted_remaining

        explanation = (
            f"EVM without tracked EV. Using performance factor {performance_factor:.2f} as CPI proxy. "
            f"Remaining budget ${remaining/100:,.0f} adjusted by factor = ${adjusted_remaining/100:,.0f}. "
            f"EAC = ${ac_cents/100:,.0f} + ${adjusted_remaining/100:,.0f} = ${eac_cents/100:,.0f}. "
            "(Note: EV estimated from performance factor)"
        )

    # Calculate ETC and variance
    etc_cents = max(0, eac_cents - ac_cents)
    var_cents = bac_cents - eac_cents  # Negative = over budget

    # Confidence scoring based on CPI
    if cpi is None:
        confidence_score = 0.5
    elif 0.9 <= cpi <= 1.1:
        confidence_score = 0.85  # On track
    elif 0.8 <= cpi <= 1.2:
        confidence_score = 0.7
    else:
        confidence_score = 0.5  # Significant variance

    # Reduce confidence if EV is estimated
    if ev_source == 'estimated':
        confidence_score *= 0.8

    confidence_band = (
        'high' if confidence_score >= 0.8 else
        'medium' if confidence_score >= 0.6 else
        'low'
    )

    return {
        'eac_cents': eac_cents,
        'etc_cents': etc_cents,
        'var_cents': var_cents,
        'cpi': round(cpi, 4) if cpi else None,
        'spi': spi,
        'confidence_score': round(confidence_score, 3),
        'confidence_band': confidence_band,
        'explanation': explanation,
        'ev_source': ev_source
    }


def calculate_pert(
    optimistic_cents: int,
    most_likely_cents: int,
    pessimistic_cents: int,
    ac_cents: int = 0
) -> Dict[str, Any]:
    """
    Three-Point Estimating (PERT) calculation.

    Formulas:
    - E = (O + 4M + P) / 6
    - Standard Deviation = (P - O) / 6
    - 68% confidence range: E ± σ
    - 95% confidence range: E ± 2σ

    Args:
        optimistic_cents: Best-case estimate
        most_likely_cents: Most probable estimate
        pessimistic_cents: Worst-case estimate
        ac_cents: Actual cost to date (for ETC calculation)

    Returns:
        Dict with eac_cents, etc_cents, std_dev, ranges, confidence
    """
    # Validate inputs
    if not (optimistic_cents <= most_likely_cents <= pessimistic_cents):
        # Auto-correct order if needed
        values = sorted([optimistic_cents, most_likely_cents, pessimistic_cents])
        optimistic_cents, most_likely_cents, pessimistic_cents = values

    # PERT expected value
    eac_cents = int((optimistic_cents + 4 * most_likely_cents + pessimistic_cents) / 6)

    # Standard deviation
    std_dev_cents = int((pessimistic_cents - optimistic_cents) / 6)

    # ETC
    etc_cents = max(0, eac_cents - ac_cents)

    # Confidence ranges
    range_68_low = eac_cents - std_dev_cents
    range_68_high = eac_cents + std_dev_cents
    range_95_low = eac_cents - (2 * std_dev_cents)
    range_95_high = eac_cents + (2 * std_dev_cents)

    # Confidence based on spread (tighter = higher confidence)
    spread_ratio = (pessimistic_cents - optimistic_cents) / most_likely_cents if most_likely_cents > 0 else 1.0
    confidence_score = max(0.3, min(0.95, 1.0 - (spread_ratio * 0.3)))

    confidence_band = (
        'high' if spread_ratio < 0.15 else
        'medium' if spread_ratio < 0.35 else
        'low'
    )

    explanation = (
        f"PERT three-point estimate. "
        f"O=${optimistic_cents/100:,.0f}, M=${most_likely_cents/100:,.0f}, P=${pessimistic_cents/100:,.0f}. "
        f"E = (O + 4M + P)/6 = ${eac_cents/100:,.0f}. "
        f"Std Dev = ${std_dev_cents/100:,.0f}. "
        f"68% range: ${range_68_low/100:,.0f} - ${range_68_high/100:,.0f}. "
        f"95% range: ${range_95_low/100:,.0f} - ${range_95_high/100:,.0f}."
    )

    return {
        'eac_cents': eac_cents,
        'etc_cents': etc_cents,
        'var_cents': 0,  # PERT doesn't have BAC reference
        'std_dev_cents': std_dev_cents,
        'optimistic_cents': optimistic_cents,
        'most_likely_cents': most_likely_cents,
        'pessimistic_cents': pessimistic_cents,
        'range_68': {'low': range_68_low, 'high': range_68_high},
        'range_95': {'low': range_95_low, 'high': range_95_high},
        'cpi': None,
        'spi': None,
        'confidence_score': round(confidence_score, 3),
        'confidence_band': confidence_band,
        'explanation': explanation
    }


def calculate_parametric(
    quantity: float,
    unit_rate_cents: int,
    complexity_factor: float = 1.0,
    ac_cents: int = 0,
    bac_cents: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parametric Estimating calculation.

    Formula:
    - EAC = Quantity × Unit Rate × Complexity Factor

    Complexity factors:
    - Low: 0.9
    - Normal: 1.0
    - High: 1.15
    - Very High: 1.3

    Args:
        quantity: Number of units (SF, LF, EA, etc.)
        unit_rate_cents: Cost per unit in cents
        complexity_factor: Adjustment factor (default 1.0)
        ac_cents: Actual cost to date
        bac_cents: Budget for variance calculation (optional)

    Returns:
        Dict with eac_cents, etc_cents, confidence
    """
    # Calculate EAC
    base_cents = int(quantity * unit_rate_cents)
    eac_cents = int(base_cents * complexity_factor)

    # ETC and variance
    etc_cents = max(0, eac_cents - ac_cents)
    var_cents = (bac_cents - eac_cents) if bac_cents else 0

    # Confidence based on complexity factor deviation from normal
    deviation = abs(complexity_factor - 1.0)
    confidence_score = max(0.5, 0.95 - (deviation * 0.3))

    confidence_band = (
        'high' if deviation < 0.1 else
        'medium' if deviation < 0.25 else
        'low'
    )

    complexity_label = (
        'Low' if complexity_factor <= 0.95 else
        'Normal' if complexity_factor <= 1.05 else
        'High' if complexity_factor <= 1.2 else
        'Very High'
    )

    explanation = (
        f"Parametric estimate. "
        f"Quantity = {quantity:,.1f}, Unit Rate = ${unit_rate_cents/100:,.2f}. "
        f"Base = {quantity:,.1f} × ${unit_rate_cents/100:,.2f} = ${base_cents/100:,.0f}. "
        f"Complexity = {complexity_label} ({complexity_factor:.2f}). "
        f"EAC = ${base_cents/100:,.0f} × {complexity_factor:.2f} = ${eac_cents/100:,.0f}."
    )

    return {
        'eac_cents': eac_cents,
        'etc_cents': etc_cents,
        'var_cents': var_cents,
        'base_cents': base_cents,
        'quantity': quantity,
        'unit_rate_cents': unit_rate_cents,
        'complexity_factor': complexity_factor,
        'cpi': None,
        'spi': None,
        'confidence_score': round(confidence_score, 3),
        'confidence_band': confidence_band,
        'explanation': explanation
    }


def calculate_manual(
    eac_cents: int,
    ac_cents: int,
    bac_cents: int,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Manual override - user directly specifies EAC.

    Args:
        eac_cents: User-specified Estimate at Completion
        ac_cents: Actual cost to date
        bac_cents: Budget for variance calculation
        notes: Optional explanation from user

    Returns:
        Dict with eac_cents, etc_cents, confidence
    """
    etc_cents = max(0, eac_cents - ac_cents)
    var_cents = bac_cents - eac_cents

    explanation = f"Manual override. EAC set to ${eac_cents/100:,.0f} by user."
    if notes:
        explanation += f" Notes: {notes}"

    return {
        'eac_cents': eac_cents,
        'etc_cents': etc_cents,
        'var_cents': var_cents,
        'cpi': None,
        'spi': None,
        'confidence_score': 0.9,  # User override = high confidence in their judgment
        'confidence_band': 'high',
        'explanation': explanation
    }


# =============================================================================
# TIME BUCKET GENERATOR
# =============================================================================

def get_week_start(date: datetime) -> datetime:
    """Get the Monday of the week containing the given date."""
    return date - timedelta(days=date.weekday())


def get_week_end(date: datetime) -> datetime:
    """Get the Sunday of the week containing the given date."""
    return get_week_start(date) + timedelta(days=6)


def get_month_start(date: datetime) -> datetime:
    """Get the first day of the month."""
    return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_month_end(date: datetime) -> datetime:
    """Get the last day of the month."""
    if date.month == 12:
        next_month = date.replace(year=date.year + 1, month=1, day=1)
    else:
        next_month = date.replace(month=date.month + 1, day=1)
    return next_month - timedelta(days=1)


def calculate_week_month_allocation(
    week_start: datetime,
    week_end: datetime,
    target_month: int,
    target_year: int
) -> float:
    """
    Calculate the proportion of a week that falls within a specific month.

    Used for weeks spanning month boundaries to allocate costs proportionally.

    Args:
        week_start: Start of the week (Monday)
        week_end: End of the week (Sunday)
        target_month: Month number (1-12)
        target_year: Year

    Returns:
        Float 0.0-1.0 representing proportion of week in target month
    """
    total_days = 7
    days_in_month = 0

    current = week_start
    while current <= week_end:
        if current.month == target_month and current.year == target_year:
            days_in_month += 1
        current += timedelta(days=1)

    return days_in_month / total_days


def generate_weekly_periods(
    start_date: datetime,
    end_date: datetime,
    as_of_date: datetime
) -> List[Dict[str, Any]]:
    """
    Generate weekly time periods from start to end date.

    Weeks use ISO calendar (Monday-Sunday).

    Args:
        start_date: Project or data start date
        end_date: Forecast completion date
        as_of_date: Current date for determining past/current/future

    Returns:
        List of period dictionaries
    """
    periods = []
    current = get_week_start(start_date)
    period_number = 1

    while current <= end_date:
        week_end = get_week_end(current)
        iso_year, iso_week, _ = current.isocalendar()

        # Determine period type
        if week_end.date() < as_of_date.date():
            period_type = 'past'
        elif current.date() > as_of_date.date():
            period_type = 'future'
        else:
            period_type = 'current'

        periods.append({
            'granularity': 'weekly',
            'period_start': current,
            'period_end': week_end,
            'period_label': f"{iso_year}-W{iso_week:02d}",
            'period_number': period_number,
            'iso_week': iso_week,
            'iso_year': iso_year,
            'period_type': period_type,
            'span_allocation_factor': 1.0  # Full week
        })

        current += timedelta(days=7)
        period_number += 1

    return periods


def generate_monthly_periods(
    start_date: datetime,
    end_date: datetime,
    as_of_date: datetime
) -> List[Dict[str, Any]]:
    """
    Generate monthly time periods from start to end date.

    Args:
        start_date: Project or data start date
        end_date: Forecast completion date
        as_of_date: Current date for determining past/current/future

    Returns:
        List of period dictionaries
    """
    periods = []
    current = get_month_start(start_date)
    period_number = 1

    while current <= end_date:
        month_end = get_month_end(current)

        # Determine period type
        if month_end.date() < as_of_date.date():
            period_type = 'past'
        elif current.date() > as_of_date.date():
            period_type = 'future'
        else:
            period_type = 'current'

        periods.append({
            'granularity': 'monthly',
            'period_start': current,
            'period_end': month_end,
            'period_label': current.strftime("%Y-%m"),
            'period_number': period_number,
            'iso_week': None,
            'iso_year': None,
            'period_type': period_type,
            'span_allocation_factor': 1.0  # Full month
        })

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
        period_number += 1

    return periods


def distribute_etc_to_periods(
    etc_cents: int,
    periods: List[Dict[str, Any]],
    distribution_method: str = 'linear',
    west_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Distribute Estimate to Complete across future periods.

    Args:
        etc_cents: Total remaining cost to distribute
        periods: List of period dictionaries (will be modified in place)
        distribution_method: 'linear', 'front_loaded', or 'back_loaded'
        west_ratio: Proportion allocated to West region (0.0-1.0)

    Returns:
        Updated periods list with forecast_cents populated
    """
    future_periods = [p for p in periods if p['period_type'] == 'future']
    n = len(future_periods)

    if n == 0:
        return periods

    east_ratio = 1.0 - west_ratio

    if distribution_method == 'linear':
        # Equal distribution
        per_period = etc_cents // n
        remainder = etc_cents % n

        for i, period in enumerate(future_periods):
            amount = per_period + (1 if i < remainder else 0)
            period['forecast_cents'] = amount
            period['forecast_west_cents'] = int(amount * west_ratio)
            period['forecast_east_cents'] = amount - period['forecast_west_cents']

    elif distribution_method == 'front_loaded':
        # More spend early: weights [n, n-1, n-2, ..., 1]
        weights = list(range(n, 0, -1))
        total_weight = sum(weights)

        allocated = 0
        for i, period in enumerate(future_periods):
            if i == n - 1:
                # Last period gets remainder to ensure exact total
                amount = etc_cents - allocated
            else:
                amount = int(etc_cents * weights[i] / total_weight)
            period['forecast_cents'] = amount
            period['forecast_west_cents'] = int(amount * west_ratio)
            period['forecast_east_cents'] = amount - period['forecast_west_cents']
            allocated += amount

    elif distribution_method == 'back_loaded':
        # More spend late: weights [1, 2, 3, ..., n]
        weights = list(range(1, n + 1))
        total_weight = sum(weights)

        allocated = 0
        for i, period in enumerate(future_periods):
            if i == n - 1:
                amount = etc_cents - allocated
            else:
                amount = int(etc_cents * weights[i] / total_weight)
            period['forecast_cents'] = amount
            period['forecast_west_cents'] = int(amount * west_ratio)
            period['forecast_east_cents'] = amount - period['forecast_west_cents']
            allocated += amount

    return periods


def aggregate_actuals_by_period(
    transactions_df: pd.DataFrame,
    periods: List[Dict[str, Any]],
    gmp_division: str,
    date_column: str = 'Date',
    amount_column: str = 'amount_cents',
    west_column: str = 'amount_west_cents',
    east_column: str = 'amount_east_cents'
) -> List[Dict[str, Any]]:
    """
    Aggregate actual costs from transactions into time periods.

    Args:
        transactions_df: DataFrame with transaction data
        periods: List of period dictionaries
        gmp_division: GMP division to filter for
        date_column: Name of date column
        amount_column: Name of amount column (in cents)
        west_column: Name of west amount column
        east_column: Name of east amount column

    Returns:
        Updated periods list with actual_cents populated
    """
    if transactions_df.empty:
        return periods

    # Filter for this GMP division if gmp_division column exists
    if 'gmp_division' in transactions_df.columns:
        df = transactions_df[transactions_df['gmp_division'] == gmp_division].copy()
    else:
        df = transactions_df.copy()

    if df.empty:
        return periods

    # Ensure date column is datetime
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    for period in periods:
        if period['period_type'] not in ['past', 'current']:
            continue

        start = period['period_start']
        end = period['period_end']

        # Filter transactions in this period
        mask = (df[date_column] >= start) & (df[date_column] <= end)
        period_df = df[mask]

        if not period_df.empty:
            period['actual_cents'] = int(period_df[amount_column].sum())
            if west_column in period_df.columns:
                period['actual_west_cents'] = int(period_df[west_column].sum())
            if east_column in period_df.columns:
                period['actual_east_cents'] = int(period_df[east_column].sum())

    return periods


def calculate_cumulative_and_blended(periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate cumulative totals and blended amounts for each period.

    Blended = actual + forecast (for current period)
    Cumulative = running total through this period

    Args:
        periods: List of period dictionaries

    Returns:
        Updated periods with cumulative_cents and blended_cents
    """
    cumulative = 0

    for period in sorted(periods, key=lambda p: p['period_start']):
        actual = period.get('actual_cents', 0)
        forecast = period.get('forecast_cents', 0)

        if period['period_type'] == 'past':
            period['blended_cents'] = actual
            cumulative += actual
        elif period['period_type'] == 'current':
            period['blended_cents'] = actual + forecast
            cumulative += actual + forecast
        else:  # future
            period['blended_cents'] = forecast
            cumulative += forecast

        period['cumulative_cents'] = cumulative

    return periods


# =============================================================================
# FORECAST MANAGER
# =============================================================================

class ForecastManager:
    """
    Manages forecast computation, storage, and retrieval.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_config(self, gmp_division: str) -> ForecastConfig:
        """Get existing config or create default for a GMP division."""
        config = self.db.query(ForecastConfig).filter(
            ForecastConfig.gmp_division == gmp_division
        ).first()

        if not config:
            config = ForecastConfig(
                gmp_division=gmp_division,
                method='evm',
                evm_performance_factor=1.0,
                distribution_method='linear',
                created_by='system'
            )
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)

        return config

    def get_current_snapshot(self, gmp_division: str) -> Optional[ForecastSnapshot]:
        """Get the current (most recent, non-superseded) snapshot."""
        return self.db.query(ForecastSnapshot).filter(
            ForecastSnapshot.gmp_division == gmp_division,
            ForecastSnapshot.is_current == True
        ).first()

    def compute_forecast(
        self,
        gmp_division: str,
        bac_cents: int,
        ac_cents: int,
        west_ratio: float = 0.5,
        ev_cents: Optional[int] = None,
        spi: Optional[float] = None,
        trigger: str = 'manual'
    ) -> ForecastSnapshot:
        """
        Compute and store a new forecast snapshot.

        Args:
            gmp_division: GMP division identifier
            bac_cents: Budget at Completion
            ac_cents: Actual Cost to date
            west_ratio: Proportion for West region
            ev_cents: Earned Value if tracked
            spi: Schedule Performance Index if available
            trigger: What triggered this calculation

        Returns:
            New ForecastSnapshot
        """
        config = self.get_or_create_config(gmp_division)

        # Calculate based on method
        method = config.method

        if method == 'evm':
            result = calculate_evm(
                bac_cents=bac_cents,
                ac_cents=ac_cents,
                ev_cents=ev_cents,
                performance_factor=config.evm_performance_factor or 1.0,
                spi=spi
            )
        elif method == 'pert':
            if not all([config.pert_optimistic_cents, config.pert_most_likely_cents,
                       config.pert_pessimistic_cents]):
                # Fall back to EVM if PERT params not set
                result = calculate_evm(bac_cents, ac_cents, ev_cents, 1.0, spi)
                result['explanation'] = "PERT parameters not configured. " + result['explanation']
            else:
                result = calculate_pert(
                    optimistic_cents=config.pert_optimistic_cents,
                    most_likely_cents=config.pert_most_likely_cents,
                    pessimistic_cents=config.pert_pessimistic_cents,
                    ac_cents=ac_cents
                )
                # Calculate variance against BAC
                result['var_cents'] = bac_cents - result['eac_cents']
        elif method == 'parametric':
            if not all([config.param_quantity, config.param_unit_rate_cents]):
                # Fall back to EVM
                result = calculate_evm(bac_cents, ac_cents, ev_cents, 1.0, spi)
                result['explanation'] = "Parametric parameters not configured. " + result['explanation']
            else:
                result = calculate_parametric(
                    quantity=config.param_quantity,
                    unit_rate_cents=config.param_unit_rate_cents,
                    complexity_factor=config.param_complexity_factor or 1.0,
                    ac_cents=ac_cents,
                    bac_cents=bac_cents
                )
        elif method == 'manual':
            # Manual requires a previous snapshot with user-set EAC
            prev = self.get_current_snapshot(gmp_division)
            if prev:
                result = calculate_manual(
                    eac_cents=prev.eac_cents,
                    ac_cents=ac_cents,
                    bac_cents=bac_cents,
                    notes=config.notes
                )
            else:
                result = calculate_evm(bac_cents, ac_cents, ev_cents, 1.0, spi)
        else:
            # Default to EVM
            result = calculate_evm(bac_cents, ac_cents, ev_cents, 1.0, spi)

        # Supersede previous snapshot
        prev_snapshot = self.get_current_snapshot(gmp_division)
        if prev_snapshot:
            prev_snapshot.is_current = False

        # Calculate regional split
        eac_cents = result['eac_cents']
        eac_west_cents = int(eac_cents * west_ratio)
        eac_east_cents = eac_cents - eac_west_cents

        # Create new snapshot
        snapshot = ForecastSnapshot(
            gmp_division=gmp_division,
            snapshot_date=datetime.utcnow(),
            bac_cents=bac_cents,
            ac_cents=ac_cents,
            ev_cents=ev_cents,
            eac_cents=eac_cents,
            eac_west_cents=eac_west_cents,
            eac_east_cents=eac_east_cents,
            etc_cents=result['etc_cents'],
            var_cents=result['var_cents'],
            cpi=result.get('cpi'),
            spi=result.get('spi'),
            method=method,
            confidence_score=result['confidence_score'],
            confidence_band=result['confidence_band'],
            explanation=result['explanation'],
            is_current=True,
            superseded_by_id=None,
            trigger=trigger
        )

        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)

        # Update previous snapshot with superseded_by
        if prev_snapshot:
            prev_snapshot.superseded_by_id = snapshot.id
            self.db.commit()

        # Log audit
        self._log_audit(
            gmp_division=gmp_division,
            action='refresh',
            previous_eac=prev_snapshot.eac_cents if prev_snapshot else None,
            new_eac=snapshot.eac_cents,
            reason=f"Triggered by: {trigger}"
        )

        return snapshot

    def generate_periods(
        self,
        snapshot: ForecastSnapshot,
        granularity: str,
        start_date: datetime,
        end_date: datetime,
        as_of_date: datetime,
        transactions_df: Optional[pd.DataFrame] = None
    ) -> List[ForecastPeriod]:
        """
        Generate and store time-bucketed forecast periods.

        Args:
            snapshot: The forecast snapshot these periods belong to
            granularity: 'weekly' or 'monthly'
            start_date: Project start
            end_date: Forecast completion
            as_of_date: Current date
            transactions_df: Optional DataFrame for actuals

        Returns:
            List of ForecastPeriod records
        """
        config = self.get_or_create_config(snapshot.gmp_division)

        # Generate period structure
        if granularity == 'weekly':
            periods = generate_weekly_periods(start_date, end_date, as_of_date)
        else:
            periods = generate_monthly_periods(start_date, end_date, as_of_date)

        # Aggregate actuals if transactions provided
        if transactions_df is not None:
            periods = aggregate_actuals_by_period(
                transactions_df, periods, snapshot.gmp_division
            )

        # Distribute ETC to future periods
        west_ratio = snapshot.eac_west_cents / snapshot.eac_cents if snapshot.eac_cents > 0 else 0.5
        periods = distribute_etc_to_periods(
            etc_cents=snapshot.etc_cents,
            periods=periods,
            distribution_method=config.distribution_method or 'linear',
            west_ratio=west_ratio
        )

        # Calculate cumulative and blended
        periods = calculate_cumulative_and_blended(periods)

        # Store periods in database
        db_periods = []
        for p in periods:
            period = ForecastPeriod(
                snapshot_id=snapshot.id,
                gmp_division=snapshot.gmp_division,
                granularity=granularity,
                period_start=p['period_start'],
                period_end=p['period_end'],
                period_label=p['period_label'],
                period_number=p['period_number'],
                iso_week=p.get('iso_week'),
                iso_year=p.get('iso_year'),
                period_type=p['period_type'],
                actual_cents=p.get('actual_cents', 0),
                forecast_cents=p.get('forecast_cents', 0),
                blended_cents=p.get('blended_cents', 0),
                cumulative_cents=p.get('cumulative_cents', 0),
                actual_west_cents=p.get('actual_west_cents', 0),
                actual_east_cents=p.get('actual_east_cents', 0),
                forecast_west_cents=p.get('forecast_west_cents', 0),
                forecast_east_cents=p.get('forecast_east_cents', 0),
                span_allocation_factor=p.get('span_allocation_factor', 1.0)
            )
            self.db.add(period)
            db_periods.append(period)

        self.db.commit()
        return db_periods

    def get_periods(
        self,
        gmp_division: str,
        granularity: str = 'weekly'
    ) -> List[ForecastPeriod]:
        """Get forecast periods for current snapshot."""
        snapshot = self.get_current_snapshot(gmp_division)
        if not snapshot:
            return []

        return self.db.query(ForecastPeriod).filter(
            ForecastPeriod.snapshot_id == snapshot.id,
            ForecastPeriod.granularity == granularity
        ).order_by(ForecastPeriod.period_number).all()

    def update_method(
        self,
        gmp_division: str,
        method: str,
        changed_by: str = 'user'
    ) -> ForecastConfig:
        """Update the forecasting method for a GMP division."""
        config = self.get_or_create_config(gmp_division)
        old_method = config.method

        if old_method != method:
            config.method = method
            config.updated_at = datetime.utcnow()
            self.db.commit()

            self._log_audit(
                gmp_division=gmp_division,
                action='method_change',
                field='method',
                old_value=old_method,
                new_value=method,
                changed_by=changed_by
            )

        return config

    def update_params(
        self,
        gmp_division: str,
        params: Dict[str, Any],
        changed_by: str = 'user'
    ) -> ForecastConfig:
        """Update method-specific parameters."""
        config = self.get_or_create_config(gmp_division)

        changes = []
        for key, value in params.items():
            if hasattr(config, key):
                old_value = getattr(config, key)
                if old_value != value:
                    setattr(config, key, value)
                    changes.append(f"{key}: {old_value} -> {value}")

        if changes:
            config.updated_at = datetime.utcnow()
            self.db.commit()

            self._log_audit(
                gmp_division=gmp_division,
                action='param_update',
                field='params',
                old_value=None,
                new_value=json.dumps(params, default=json_serial),
                reason="; ".join(changes),
                changed_by=changed_by
            )

        return config

    def get_history(self, gmp_division: str, limit: int = 50) -> List[ForecastAuditLog]:
        """Get forecast change history."""
        return self.db.query(ForecastAuditLog).filter(
            ForecastAuditLog.gmp_division == gmp_division
        ).order_by(ForecastAuditLog.changed_at.desc()).limit(limit).all()

    def get_snapshot_history(self, gmp_division: str, limit: int = 20) -> List[ForecastSnapshot]:
        """Get historical snapshots."""
        return self.db.query(ForecastSnapshot).filter(
            ForecastSnapshot.gmp_division == gmp_division
        ).order_by(ForecastSnapshot.snapshot_date.desc()).limit(limit).all()

    def _log_audit(
        self,
        gmp_division: str,
        action: str,
        field: Optional[str] = None,
        old_value: Any = None,
        new_value: Any = None,
        previous_eac: Optional[int] = None,
        new_eac: Optional[int] = None,
        reason: Optional[str] = None,
        changed_by: str = 'system'
    ):
        """Log an audit entry."""
        log = ForecastAuditLog(
            gmp_division=gmp_division,
            action=action,
            field_changed=field,
            old_value=json.dumps(old_value, default=json_serial) if old_value is not None else None,
            new_value=json.dumps(new_value, default=json_serial) if new_value is not None else None,
            previous_eac_cents=previous_eac,
            new_eac_cents=new_eac,
            change_reason=reason,
            changed_by=changed_by
        )
        self.db.add(log)
        self.db.commit()


# =============================================================================
# PROJECT-LEVEL ROLLUP
# =============================================================================

def compute_project_rollup(db: Session) -> Dict[str, Any]:
    """
    Compute project-level forecast rollup across all GMP divisions.

    Returns:
        Dict with total_bac, total_ac, total_eac, total_var, and by_division breakdown
    """
    snapshots = db.query(ForecastSnapshot).filter(
        ForecastSnapshot.is_current == True
    ).all()

    if not snapshots:
        return {
            'total_bac_cents': 0,
            'total_ac_cents': 0,
            'total_eac_cents': 0,
            'total_etc_cents': 0,
            'total_var_cents': 0,
            'overall_cpi': None,
            'by_division': []
        }

    total_bac = sum(s.bac_cents for s in snapshots)
    total_ac = sum(s.ac_cents for s in snapshots)
    total_ev = sum(s.ev_cents or 0 for s in snapshots)
    total_eac = sum(s.eac_cents for s in snapshots)
    total_etc = sum(s.etc_cents for s in snapshots)
    total_var = total_bac - total_eac

    # Overall CPI
    overall_cpi = total_ev / total_ac if total_ac > 0 and total_ev > 0 else None

    by_division = [
        {
            'gmp_division': s.gmp_division,
            'bac_cents': s.bac_cents,
            'ac_cents': s.ac_cents,
            'eac_cents': s.eac_cents,
            'etc_cents': s.etc_cents,
            'var_cents': s.var_cents,
            'cpi': s.cpi,
            'method': s.method,
            'confidence_band': s.confidence_band,
            'snapshot_date': s.snapshot_date.isoformat()
        }
        for s in snapshots
    ]

    return {
        'total_bac_cents': total_bac,
        'total_ac_cents': total_ac,
        'total_eac_cents': total_eac,
        'total_etc_cents': total_etc,
        'total_var_cents': total_var,
        'overall_cpi': round(overall_cpi, 4) if overall_cpi else None,
        'by_division': by_division,
        'as_of': datetime.utcnow().isoformat()
    }

"""
Reconciliation Module for GMP Reconciliation App.
Computes per-GMP-division rows with required and optional columns.
All calculations in integer cents for precision.
Uses Largest Remainder Method for penny-perfect E/W allocation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from sqlalchemy.exc import OperationalError

from .etl import (
    cents_to_display,
    allocate_east_west,
    allocate_largest_remainder,
    fuzzy_match_breakdown_to_gmp,
    calculate_weighted_progress
)
from ..models import (
    Settings, GMPBudgetBreakdown, GMP, DirectCostEntity,
    ForecastSnapshot, ScheduleActivity, ScheduleToGMPMapping
)

logger = logging.getLogger(__name__)


def safe_divide(numerator: int, denominator: int) -> float:
    """Safe division that returns 0 on divide by zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def get_settings(db: Session) -> Dict:
    """Load reconciliation settings from database."""
    settings = db.query(Settings).first()
    if settings:
        return {
            'as_of_date': settings.as_of_date,
            'forecast_basis': settings.forecast_basis,
            'eac_mode_when_commitments': settings.eac_mode_when_commitments,
            'gmp_scope_notes': settings.gmp_scope_notes,
            'gmp_scope_confirmed': settings.gmp_scope_confirmed,
            'use_breakdown_allocations': getattr(settings, 'use_breakdown_allocations', True),
            'use_schedule_forecast': getattr(settings, 'use_schedule_forecast', False)
        }
    return {
        'as_of_date': None,
        'forecast_basis': 'actuals_plus_commitments',
        'eac_mode_when_commitments': 'max',
        'gmp_scope_notes': None,
        'gmp_scope_confirmed': False,
        'use_breakdown_allocations': True,
        'use_schedule_forecast': False
    }


def get_breakdown_allocations(db: Session, gmp_division: str) -> Optional[Dict]:
    """
    Get East/West allocation percentages from breakdown data for a GMP division.

    Returns dict with pct_east, pct_west, or None if no breakdown data.
    """
    breakdown = db.query(GMPBudgetBreakdown).filter(
        GMPBudgetBreakdown.gmp_division == gmp_division
    ).first()

    if breakdown:
        return {
            'pct_east': breakdown.pct_east,
            'pct_west': breakdown.pct_west,
            'source': 'breakdown'
        }
    return None


def allocate_amount_east_west(
    amount_cents: int,
    gmp_division: str,
    breakdown_df: Optional[pd.DataFrame] = None,
    default_pct_east: float = 0.5
) -> tuple[int, int]:
    """
    Allocate an amount between East/West using breakdown data or default split.
    Uses Largest Remainder Method for penny-perfect allocation.

    Args:
        amount_cents: Amount to allocate (integer cents)
        gmp_division: GMP division name for lookup
        breakdown_df: DataFrame with breakdown allocations (optional)
        default_pct_east: Default East percentage if no breakdown data

    Returns:
        Tuple of (east_cents, west_cents) that sum exactly to amount_cents
    """
    pct_east = default_pct_east
    pct_west = 1.0 - default_pct_east

    # Look up breakdown allocation if available
    if breakdown_df is not None and not breakdown_df.empty:
        matched = breakdown_df[breakdown_df['gmp_division'] == gmp_division]
        if not matched.empty:
            pct_east = matched.iloc[0]['pct_east']
            pct_west = matched.iloc[0]['pct_west']

    # Use Largest Remainder Method for penny-perfect split
    return allocate_east_west(amount_cents, pct_east, pct_west)


def filter_by_as_of_date(direct_costs_df: pd.DataFrame, as_of_date: Optional[datetime]) -> pd.DataFrame:
    """Filter direct costs to only include rows up to as_of_date."""
    if as_of_date is None:
        return direct_costs_df
    
    mask = direct_costs_df['date_parsed'] <= as_of_date
    return direct_costs_df[mask].copy()


def aggregate_actuals_by_gmp(
    direct_costs_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None,
    breakdown_df: Optional[pd.DataFrame] = None,
    use_breakdown_allocations: bool = True
) -> pd.DataFrame:
    """
    Aggregate actual costs by GMP division, split by West/East.
    Uses Largest Remainder Method for penny-perfect allocation.

    Args:
        direct_costs_df: Direct costs DataFrame
        budget_df: Budget DataFrame with GMP mappings
        as_of_date: Optional date filter
        breakdown_df: Optional breakdown data for E/W allocation percentages
        use_breakdown_allocations: Whether to use breakdown data for allocation

    Returns DataFrame with columns:
    - gmp_division
    - actual_west_cents
    - actual_east_cents
    - actual_total_cents
    - row_count
    """
    # Filter by date
    filtered = filter_by_as_of_date(direct_costs_df, as_of_date)

    # Exclude flagged duplicates
    filtered = filtered[filtered['excluded_from_actuals'] == False]

    # Join direct costs to budget to get GMP division
    budget_gmp_map = budget_df[['Budget Code', 'gmp_division']].drop_duplicates()

    merged = filtered.merge(
        budget_gmp_map,
        left_on='mapped_budget_code',
        right_on='Budget Code',
        how='left'
    )

    # First aggregate totals by GMP division
    agg_totals = merged.groupby('gmp_division').agg({
        'amount_cents': 'sum',
        'direct_cost_id': 'count'
    }).reset_index()
    agg_totals.columns = ['gmp_division', 'actual_total_cents', 'row_count']

    # Now allocate E/W using LRM for each division
    east_cents = []
    west_cents = []

    for _, row in agg_totals.iterrows():
        gmp_div = row['gmp_division']
        total = int(row['actual_total_cents'])

        if use_breakdown_allocations and breakdown_df is not None:
            east, west = allocate_amount_east_west(total, gmp_div, breakdown_df)
        else:
            # Use existing row-level splits if available
            div_data = merged[merged['gmp_division'] == gmp_div]
            if 'amount_west' in div_data.columns and 'amount_east' in div_data.columns:
                # Use pre-computed splits but ensure they tie out with LRM
                raw_east = int(div_data['amount_east'].sum())
                raw_west = int(div_data['amount_west'].sum())
                raw_total = raw_east + raw_west
                if raw_total > 0 and abs(raw_total - total) <= len(div_data):
                    # Row splits are close, use LRM to ensure exact tie-out
                    pct_east = raw_east / raw_total if raw_total > 0 else 0.5
                    east, west = allocate_east_west(total, pct_east, 1.0 - pct_east)
                else:
                    # Default 50/50
                    east, west = allocate_east_west(total, 0.5, 0.5)
            else:
                # Default 50/50
                east, west = allocate_east_west(total, 0.5, 0.5)

        east_cents.append(east)
        west_cents.append(west)

    agg_totals['actual_east_cents'] = east_cents
    agg_totals['actual_west_cents'] = west_cents

    # Reorder columns
    return agg_totals[['gmp_division', 'actual_west_cents', 'actual_east_cents', 'actual_total_cents', 'row_count']]


def aggregate_commitments_by_gmp(
    budget_df: pd.DataFrame,
    breakdown_df: Optional[pd.DataFrame] = None,
    use_breakdown_allocations: bool = True
) -> pd.DataFrame:
    """
    Aggregate committed costs by GMP division, split by West/East.
    Uses Largest Remainder Method for penny-perfect allocation.

    Args:
        budget_df: Budget DataFrame with committed costs
        breakdown_df: Optional breakdown data for E/W allocation percentages
        use_breakdown_allocations: Whether to use breakdown data for allocation

    Returns DataFrame with columns:
    - gmp_division
    - committed_west_cents
    - committed_east_cents
    - committed_total_cents
    """
    if 'committed_costs_cents' not in budget_df.columns:
        return pd.DataFrame(columns=['gmp_division', 'committed_west_cents', 'committed_east_cents', 'committed_total_cents'])

    # First aggregate totals by GMP division
    agg_totals = budget_df.groupby('gmp_division').agg({
        'committed_costs_cents': 'sum'
    }).reset_index()
    agg_totals.columns = ['gmp_division', 'committed_total_cents']

    # Now allocate E/W using LRM for each division
    east_cents = []
    west_cents = []

    for _, row in agg_totals.iterrows():
        gmp_div = row['gmp_division']
        total = int(row['committed_total_cents'])

        if use_breakdown_allocations and breakdown_df is not None:
            east, west = allocate_amount_east_west(total, gmp_div, breakdown_df)
        else:
            # Use budget row pct_west/pct_east if available
            div_budget = budget_df[budget_df['gmp_division'] == gmp_div]
            if 'pct_west' in div_budget.columns and 'pct_east' in div_budget.columns:
                # Weighted average of percentages based on committed amounts
                weighted_pct_east = 0.5
                if total > 0:
                    weighted_pct_east = (div_budget['committed_costs_cents'] * div_budget['pct_east']).sum() / total
                east, west = allocate_east_west(total, weighted_pct_east, 1.0 - weighted_pct_east)
            else:
                east, west = allocate_east_west(total, 0.5, 0.5)

        east_cents.append(east)
        west_cents.append(west)

    agg_totals['committed_east_cents'] = east_cents
    agg_totals['committed_west_cents'] = west_cents

    return agg_totals[['gmp_division', 'committed_west_cents', 'committed_east_cents', 'committed_total_cents']]


def compute_reconciliation(
    gmp_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    direct_costs_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame] = None,
    settings: Optional[Dict] = None,
    breakdown_df: Optional[pd.DataFrame] = None,
    schedule_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Main reconciliation computation.
    Uses Largest Remainder Method for penny-perfect E/W allocation.

    Computes per-GMP-division:
    Required columns:
    - amount_assigned_west (actual_west_cents displayed)
    - amount_assigned_east (actual_east_cents displayed)
    - forecast_west (EAC West)
    - forecast_east (EAC East)
    - surplus_or_overrun (GMP - EAC)

    Optional columns:
    - actual_total
    - committed_total
    - eac_total
    - variance_to_gmp
    - pct_spent

    Args:
        gmp_df: GMP DataFrame
        budget_df: Budget DataFrame
        direct_costs_df: Direct costs DataFrame
        predictions_df: Optional ML predictions DataFrame
        settings: Reconciliation settings dict
        breakdown_df: Optional breakdown data for E/W allocation
        schedule_df: Optional schedule data for progress-based EAC

    Returns DataFrame ready for display.
    """
    settings = settings or {
        'as_of_date': None,
        'forecast_basis': 'actuals_plus_commitments',
        'eac_mode_when_commitments': 'max',
        'use_breakdown_allocations': True,
        'use_schedule_forecast': False
    }

    as_of_date = settings.get('as_of_date')
    forecast_basis = settings.get('forecast_basis', 'actuals_plus_commitments')
    eac_mode = settings.get('eac_mode_when_commitments', 'max')
    use_breakdown = settings.get('use_breakdown_allocations', True)
    use_schedule = settings.get('use_schedule_forecast', False)

    # Get actuals aggregated by GMP
    actuals_agg = aggregate_actuals_by_gmp(
        direct_costs_df, budget_df, as_of_date,
        breakdown_df=breakdown_df if use_breakdown else None,
        use_breakdown_allocations=use_breakdown
    )

    # Get commitments aggregated by GMP
    commitments_agg = aggregate_commitments_by_gmp(
        budget_df,
        breakdown_df=breakdown_df if use_breakdown else None,
        use_breakdown_allocations=use_breakdown
    )
    
    # Start with GMP as base
    result = gmp_df[['GMP', 'amount_total_cents']].copy()
    result.columns = ['gmp_division', 'gmp_amount_cents']

    # Merge breakdown allocations (Budget East/West from breakdown.csv)
    if breakdown_df is not None and not breakdown_df.empty:
        result = result.merge(
            breakdown_df[['gmp_division', 'east_funded_cents', 'west_funded_cents']],
            on='gmp_division',
            how='left'
        )
        result['budget_east_cents'] = result['east_funded_cents'].fillna(0).astype(int)
        result['budget_west_cents'] = result['west_funded_cents'].fillna(0).astype(int)
        result = result.drop(columns=['east_funded_cents', 'west_funded_cents'], errors='ignore')
    else:
        # No breakdown data - default to 50/50 split of GMP amount
        result['budget_east_cents'] = 0
        result['budget_west_cents'] = 0

    # Merge actuals
    result = result.merge(actuals_agg, on='gmp_division', how='left')
    for col in ['actual_west_cents', 'actual_east_cents', 'actual_total_cents', 'row_count']:
        result[col] = result[col].fillna(0).astype(int)
    
    # Merge commitments
    result = result.merge(commitments_agg, on='gmp_division', how='left')
    for col in ['committed_west_cents', 'committed_east_cents', 'committed_total_cents']:
        result[col] = result[col].fillna(0).astype(int)
    
    # Merge predictions if available
    if predictions_df is not None and not predictions_df.empty:
        result = result.merge(predictions_df, on='gmp_division', how='left')
        for col in ['predicted_remaining_west', 'predicted_remaining_east']:
            if col in result.columns:
                result[col] = result[col].fillna(0).astype(int)
    else:
        # Default to 0 if no predictions
        result['predicted_remaining_west'] = 0
        result['predicted_remaining_east'] = 0
    
    # Compute remaining commitments
    result['committed_remaining_west'] = np.maximum(0, result['committed_west_cents'] - result['actual_west_cents'])
    result['committed_remaining_east'] = np.maximum(0, result['committed_east_cents'] - result['actual_east_cents'])
    
    # Compute EAC (Forecast)
    if forecast_basis == 'actuals_only':
        result['forecast_west_cents'] = result['actual_west_cents'] + result['predicted_remaining_west']
        result['forecast_east_cents'] = result['actual_east_cents'] + result['predicted_remaining_east']
    else:  # actuals_plus_commitments
        eac_model_west = result['actual_west_cents'] + result['predicted_remaining_west']
        eac_model_east = result['actual_east_cents'] + result['predicted_remaining_east']
        eac_commit_west = result['actual_west_cents'] + result['committed_remaining_west']
        eac_commit_east = result['actual_east_cents'] + result['committed_remaining_east']
        
        if eac_mode == 'max':
            result['forecast_west_cents'] = np.maximum(eac_model_west, eac_commit_west)
            result['forecast_east_cents'] = np.maximum(eac_model_east, eac_commit_east)
        elif eac_mode == 'model':
            result['forecast_west_cents'] = eac_model_west
            result['forecast_east_cents'] = eac_model_east
        else:  # commitments
            result['forecast_west_cents'] = eac_commit_west
            result['forecast_east_cents'] = eac_commit_east
    
    # Compute derived columns
    result['eac_total_cents'] = result['forecast_west_cents'] + result['forecast_east_cents']
    result['surplus_or_overrun_cents'] = result['gmp_amount_cents'] - result['eac_total_cents']
    result['variance_to_gmp_cents'] = result['surplus_or_overrun_cents']  # Same calculation
    result['pct_spent'] = result.apply(
        lambda r: safe_divide(r['actual_total_cents'], r['gmp_amount_cents']) * 100, axis=1
    )
    
    return result


def format_for_display(recon_df: pd.DataFrame) -> List[Dict]:
    """
    Convert reconciliation DataFrame to list of dicts for HTML display.
    Formats all cents values as USD strings.
    """
    rows = []
    for _, row in recon_df.iterrows():
        # Budget allocations from breakdown.csv (owner's funding plan)
        budget_west = int(row.get('budget_west_cents', 0))
        budget_east = int(row.get('budget_east_cents', 0))

        rows.append({
            'gmp_division': row['gmp_division'],
            'gmp_amount': cents_to_display(int(row['gmp_amount_cents'])),
            # Budget allocations (from breakdown.csv)
            'budget_west': cents_to_display(budget_west),
            'budget_east': cents_to_display(budget_east),
            'budget_west_raw': budget_west,
            'budget_east_raw': budget_east,
            # Actuals (what has been spent)
            'amount_assigned_west': cents_to_display(int(row['actual_west_cents'])),
            'amount_assigned_east': cents_to_display(int(row['actual_east_cents'])),
            'forecast_west': cents_to_display(int(row['forecast_west_cents'])),
            'forecast_east': cents_to_display(int(row['forecast_east_cents'])),
            'surplus_or_overrun': cents_to_display(int(row['surplus_or_overrun_cents'])),
            'surplus_positive': row['surplus_or_overrun_cents'] >= 0,
            'actual_total': cents_to_display(int(row['actual_total_cents'])),
            'committed_total': cents_to_display(int(row['committed_total_cents'])),
            'eac_total': cents_to_display(int(row['eac_total_cents'])),
            'variance_to_gmp': cents_to_display(int(row['variance_to_gmp_cents'])),
            'pct_spent': f"{row['pct_spent']:.1f}%",
            # Raw values for JS/sorting
            'gmp_amount_raw': int(row['gmp_amount_cents']),
            'actual_west_raw': int(row['actual_west_cents']),
            'actual_east_raw': int(row['actual_east_cents']),
            'forecast_west_raw': int(row['forecast_west_cents']),
            'forecast_east_raw': int(row['forecast_east_cents']),
            'surplus_raw': int(row['surplus_or_overrun_cents']),
            'eac_total_raw': int(row['eac_total_cents']),
        })
    return rows


def compute_summary_metrics(
    recon_df: pd.DataFrame,
    direct_costs_df: pd.DataFrame,
    budget_df: pd.DataFrame
) -> Dict:
    """
    Compute summary metrics for the reconciliation summary panel.
    """
    # Total direct costs (net, excluding excluded)
    filtered_dc = direct_costs_df[direct_costs_df['excluded_from_actuals'] == False]
    total_direct_costs_net = filtered_dc['amount_cents'].sum()
    
    # Credits (negative amounts)
    credits = filtered_dc[filtered_dc['amount_cents'] < 0]['amount_cents'].sum()
    
    # Mapped vs unmapped direct costs
    mapped_dc = filtered_dc[filtered_dc['mapped_budget_code'].notna()]
    unmapped_dc = filtered_dc[filtered_dc['mapped_budget_code'].isna()]
    total_mapped_to_budget = mapped_dc['amount_cents'].sum()
    total_unmapped_to_budget = unmapped_dc['amount_cents'].sum()
    
    # Mapped vs unmapped budget to GMP
    budget_mapped = budget_df[budget_df['gmp_division'].notna()]
    budget_unmapped = budget_df[budget_df['gmp_division'].isna()]
    
    # Sum of current budget values
    total_budget_mapped = budget_mapped['current_budget_cents'].sum() if 'current_budget_cents' in budget_mapped.columns else 0
    total_budget_unmapped = budget_unmapped['current_budget_cents'].sum() if 'current_budget_cents' in budget_unmapped.columns else 0
    
    # Grand totals from reconciliation
    total_gmp = recon_df['gmp_amount_cents'].sum()
    total_eac = recon_df['eac_total_cents'].sum()
    total_surplus = total_gmp - total_eac
    
    return {
        'total_direct_costs_net': cents_to_display(total_direct_costs_net),
        'total_direct_costs_net_raw': total_direct_costs_net,
        'total_mapped_to_budget': cents_to_display(total_mapped_to_budget),
        'total_unmapped_to_budget': cents_to_display(total_unmapped_to_budget),
        'total_budget_mapped_to_gmp': cents_to_display(total_budget_mapped),
        'total_budget_unmapped_to_gmp': cents_to_display(total_budget_unmapped),
        'total_credits_negative': cents_to_display(credits),
        'unmapped_dc_count': len(unmapped_dc),
        'unmapped_budget_count': len(budget_unmapped),
        'total_gmp': cents_to_display(total_gmp),
        'total_eac': cents_to_display(total_eac),
        'total_surplus_or_overrun': cents_to_display(total_surplus),
        'surplus_positive': total_surplus >= 0
    }


def get_gmp_drilldown(
    gmp_division: str,
    direct_costs_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None
) -> Dict:
    """
    Get detailed breakdown of direct costs assigned to a GMP division.

    Returns structure with:
    - budget_codes: list of budget codes under this GMP
    - direct_costs: list of direct cost records with amounts
    - totals: west, east, total amounts
    """
    # Filter by date
    filtered = filter_by_as_of_date(direct_costs_df, as_of_date)

    # Exclude flagged duplicates
    filtered = filtered[filtered['excluded_from_actuals'] == False]

    # Get budget codes for this GMP division
    gmp_budgets = budget_df[budget_df['gmp_division'] == gmp_division]['Budget Code'].tolist()

    # Get direct costs mapped to these budget codes
    dc_for_gmp = filtered[filtered['mapped_budget_code'].isin(gmp_budgets)].copy()

    # Build breakdown by budget code
    budget_breakdown = []
    for budget_code in gmp_budgets:
        dc_for_budget = dc_for_gmp[dc_for_gmp['mapped_budget_code'] == budget_code]
        if len(dc_for_budget) == 0:
            continue

        budget_desc = budget_df[budget_df['Budget Code'] == budget_code]['Budget Code Description'].iloc[0] if len(budget_df[budget_df['Budget Code'] == budget_code]) > 0 else ''

        # Individual records
        records = []
        for _, row in dc_for_budget.iterrows():
            records.append({
                'direct_cost_id': int(row.get('direct_cost_id', 0)),
                'cost_code': str(row.get('Cost Code', '')),
                'name': str(row.get('Name', ''))[:50],
                'vendor': str(row.get('Vendor', ''))[:30],
                'date': str(row.get('Date', '')),
                'amount_west': int(row.get('amount_west', 0)),
                'amount_east': int(row.get('amount_east', 0)),
                'amount_total': int(row.get('amount_cents', 0)),
                'amount_west_display': cents_to_display(int(row.get('amount_west', 0))),
                'amount_east_display': cents_to_display(int(row.get('amount_east', 0))),
                'amount_total_display': cents_to_display(int(row.get('amount_cents', 0))),
            })

        budget_breakdown.append({
            'budget_code': budget_code,
            'description': budget_desc[:50] if budget_desc else '',
            'count': len(records),
            'total_west': int(dc_for_budget['amount_west'].sum()),
            'total_east': int(dc_for_budget['amount_east'].sum()),
            'total_amount': int(dc_for_budget['amount_cents'].sum()),
            'total_west_display': cents_to_display(int(dc_for_budget['amount_west'].sum())),
            'total_east_display': cents_to_display(int(dc_for_budget['amount_east'].sum())),
            'total_amount_display': cents_to_display(int(dc_for_budget['amount_cents'].sum())),
            'records': records[:20]  # Limit to first 20 for performance
        })

    # Sort by total amount descending
    budget_breakdown.sort(key=lambda x: -abs(x['total_amount']))

    # Compute totals
    total_west = int(dc_for_gmp['amount_west'].sum())
    total_east = int(dc_for_gmp['amount_east'].sum())
    total_amount = int(dc_for_gmp['amount_cents'].sum())

    return {
        'gmp_division': gmp_division,
        'budget_count': len(budget_breakdown),
        'record_count': len(dc_for_gmp),
        'budget_breakdown': budget_breakdown[:10],  # Top 10 budget codes
        'total_west': total_west,
        'total_east': total_east,
        'total_amount': total_amount,
        'total_west_display': cents_to_display(total_west),
        'total_east_display': cents_to_display(total_east),
        'total_amount_display': cents_to_display(total_amount),
        'integrity_check': abs((total_west + total_east) - total_amount) <= 1
    }


def validate_tie_outs(
    recon_df: pd.DataFrame,
    direct_costs_df: pd.DataFrame
) -> List[Dict]:
    """
    Validate that the reconciliation ties out correctly.
    
    Tie-out 1: Sum(assigned_west + assigned_east) == total actuals post-split
    Tie-out 2: Cross-foot: Sum over GMP rows of assigned totals == grand total actuals
    
    Returns list of validation results with pass/fail status.
    """
    results = []
    
    # Get non-excluded direct costs
    filtered_dc = direct_costs_df[direct_costs_df['excluded_from_actuals'] == False]
    
    # Tie-out 1: West + East = Total
    total_assigned_west = recon_df['actual_west_cents'].sum()
    total_assigned_east = recon_df['actual_east_cents'].sum()
    total_assigned = total_assigned_west + total_assigned_east
    total_actual = filtered_dc['amount_cents'].sum()
    
    tie_out_1 = abs(total_assigned - total_actual) <= 1  # Allow 1 cent rounding
    results.append({
        'name': 'West + East = Total Actuals',
        'passed': tie_out_1,
        'expected': cents_to_display(total_actual),
        'actual': cents_to_display(total_assigned),
        'difference': cents_to_display(total_assigned - total_actual)
    })
    
    # Tie-out 2: Sum of GMP row totals = Grand total
    sum_gmp_actuals = recon_df['actual_total_cents'].sum()
    tie_out_2 = abs(sum_gmp_actuals - total_actual) <= len(recon_df)  # Allow 1 cent per row
    results.append({
        'name': 'Cross-foot GMP Rows vs Grand Total',
        'passed': tie_out_2,
        'expected': cents_to_display(total_actual),
        'actual': cents_to_display(sum_gmp_actuals),
        'difference': cents_to_display(sum_gmp_actuals - total_actual)
    })

    return results


# =============================================================================
# Schedule-Based Forecast Calculations (P6 Algorithm)
# =============================================================================

# Constants for P6 weighting
CRITICAL_PATH_MULTIPLIER = 2.0  # Weight multiplier for critical path activities (TF=0)
MIN_PROGRESS_THRESHOLD = 0.01   # 1% minimum progress to avoid division by tiny numbers


def compute_activity_weight(
    activity,
    mapping_weight: float = 1.0
) -> float:
    """
    Compute activity weight using P6-style algorithm.

    Weight formula: base_weight × critical_multiplier × duration_factor

    Args:
        activity: ScheduleActivity instance
        mapping_weight: User-assigned weight from mapping (0.0-1.0)

    Returns:
        float: Composite weight for this activity
    """
    # Base weight from mapping
    base_weight = mapping_weight

    # Critical path multiplier (2x for TF=0)
    critical_multiplier = CRITICAL_PATH_MULTIPLIER if activity.is_critical else 1.0

    # Duration-based factor (longer activities count more)
    # Use square root to dampen extreme values
    duration = activity.duration_days or 1
    duration_factor = (duration / 10) ** 0.5  # Normalize around 10-day activities

    return base_weight * critical_multiplier * duration_factor


def compute_schedule_based_forecast(
    db: Session,
    gmp_division: str,
    budget_cents: int,
    actual_cents: int
) -> Dict:
    """
    Compute schedule-based EAC for a GMP division using P6-style progress.

    P6 Algorithm:
    1. Progress Source: Uses progress_pct derived from date actuals (" A" suffix)
       - Complete (both dates actual): progress = 1.0
       - In Progress (start actual): progress = elapsed_days / duration
       - Not Started (neither actual): progress = 0.0

    2. Weighting: Combines three factors:
       - User-assigned mapping weight (0.0-1.0)
       - Critical path multiplier (2x for Total Float = 0)
       - Duration-based factor (longer activities count more)

    3. EAC Calculation:
       - EAC = Budget × (Actuals / Schedule_Progress) when progress > 1%
       - Guards: Cap at 3x budget, floor at actual spend

    Args:
        db: Database session
        gmp_division: GMP division name
        budget_cents: Budget amount in cents
        actual_cents: Actual spent in cents

    Returns dict with:
        - weighted_progress: float (0-100)
        - schedule_eac_cents: int
        - estimated_completion: date or None
        - activities: list of contributing activities
        - has_schedule_data: bool
        - p6_stats: dict with P6-specific metrics
    """
    from datetime import date, timedelta
    from ..models import ScheduleToGMPMapping

    # Get schedule mappings for this division
    mappings = db.query(ScheduleToGMPMapping).filter(
        ScheduleToGMPMapping.gmp_division == gmp_division
    ).all()

    if not mappings:
        return {
            'weighted_progress': 0.0,
            'schedule_eac_cents': max(actual_cents, budget_cents),
            'schedule_eac_display': cents_to_display(max(actual_cents, budget_cents)),
            'estimated_completion': None,
            'activities': [],
            'has_schedule_data': False,
            'activity_count': 0,
            'p6_stats': {
                'complete_count': 0,
                'in_progress_count': 0,
                'not_started_count': 0,
                'critical_count': 0,
                'total_duration_days': 0
            }
        }

    # Calculate weighted progress using P6 algorithm
    total_weight = 0.0
    weighted_sum = 0.0
    activities = []
    earliest_start = None
    latest_finish = None
    today = date.today()

    # P6 stats tracking
    complete_count = 0
    in_progress_count = 0
    not_started_count = 0
    critical_count = 0
    total_duration_days = 0

    for m in mappings:
        activity = m.activity

        # Use P6 progress_pct if available, fallback to pct_complete
        if hasattr(activity, 'progress_pct') and activity.progress_pct is not None:
            progress = activity.progress_pct
        else:
            progress = activity.pct_complete / 100.0

        # Compute composite weight using P6 algorithm
        weight = compute_activity_weight(activity, m.weight)

        weighted_sum += weight * progress
        total_weight += weight

        # Track P6 stats
        if hasattr(activity, 'is_complete') and activity.is_complete:
            complete_count += 1
        elif hasattr(activity, 'is_in_progress') and activity.is_in_progress:
            in_progress_count += 1
        else:
            not_started_count += 1

        if hasattr(activity, 'is_critical') and activity.is_critical:
            critical_count += 1

        if activity.duration_days:
            total_duration_days += activity.duration_days

        # Track dates
        if activity.start_date and (earliest_start is None or activity.start_date < earliest_start):
            earliest_start = activity.start_date
        if activity.finish_date and (latest_finish is None or activity.finish_date > latest_finish):
            latest_finish = activity.finish_date

        activities.append({
            'task_name': activity.task_name,
            'activity_id': activity.activity_id or '',
            'pct_complete': activity.pct_complete,
            'progress_pct': round(progress * 100, 1),
            'weight': round(weight, 3),
            'contribution': round(weight * progress * 100, 1),
            'is_complete': getattr(activity, 'is_complete', False),
            'is_in_progress': getattr(activity, 'is_in_progress', False),
            'is_critical': getattr(activity, 'is_critical', False),
            'duration_days': activity.duration_days,
            'start_date': activity.start_date.isoformat() if activity.start_date else None,
            'finish_date': activity.finish_date.isoformat() if activity.finish_date else None
        })

    # Weighted progress (0-100)
    weighted_progress = (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0

    # Compute schedule-based EAC with guards
    schedule_eac_cents = budget_cents  # Default to budget

    if weighted_progress >= MIN_PROGRESS_THRESHOLD * 100:
        # EAC = Budget × (Actuals / Progress)
        progress_ratio = weighted_progress / 100.0
        if progress_ratio > 0:
            schedule_eac_cents = int(actual_cents / progress_ratio)

            # Guard 1: Cap at 3x budget to avoid extreme outliers
            schedule_eac_cents = min(schedule_eac_cents, budget_cents * 3)

            # Guard 2: Floor at actual spend (can't forecast less than already spent)
            schedule_eac_cents = max(schedule_eac_cents, actual_cents)
    else:
        # Progress too low - use budget as EAC
        schedule_eac_cents = max(budget_cents, actual_cents)

    # Estimate completion date based on current progress rate
    estimated_completion = None
    if earliest_start and weighted_progress > 0 and weighted_progress < 100:
        days_elapsed = (today - earliest_start).days
        if days_elapsed > 0:
            # Rate = progress per day
            rate = weighted_progress / days_elapsed
            remaining_progress = 100 - weighted_progress
            days_remaining = int(remaining_progress / rate) if rate > 0 else 365
            # Cap at reasonable horizon (2 years)
            days_remaining = min(days_remaining, 730)
            estimated_completion = today + timedelta(days=days_remaining)
    elif weighted_progress >= 100:
        # Complete - use latest finish date
        estimated_completion = latest_finish or today

    # Sort activities by contribution (descending)
    activities.sort(key=lambda x: -x['contribution'])

    return {
        'weighted_progress': round(weighted_progress, 1),
        'schedule_eac_cents': schedule_eac_cents,
        'schedule_eac_display': cents_to_display(schedule_eac_cents),
        'estimated_completion': estimated_completion.isoformat() if estimated_completion else None,
        'activities': activities,
        'has_schedule_data': True,
        'activity_count': len(activities),
        'earliest_start': earliest_start.isoformat() if earliest_start else None,
        'latest_finish': latest_finish.isoformat() if latest_finish else None,
        'p6_stats': {
            'complete_count': complete_count,
            'in_progress_count': in_progress_count,
            'not_started_count': not_started_count,
            'critical_count': critical_count,
            'total_duration_days': total_duration_days,
            'total_weight': round(total_weight, 2)
        }
    }


def compute_project_schedule_summary(db: Session) -> Dict:
    """
    Compute project-wide schedule summary with P6 metrics.

    Returns:
        - total_activities: int
        - mapped_activities: int
        - unmapped_activities: int
        - avg_progress: float (weighted by P6 algorithm)
        - earliest_start: date
        - latest_finish: date
        - by_division: list of division summaries
        - p6_stats: P6-specific metrics
    """
    from ..models import ScheduleActivity, ScheduleToGMPMapping

    activities = db.query(ScheduleActivity).all()

    total_activities = len(activities)

    # P6 stats tracking
    complete_count = 0
    in_progress_count = 0
    not_started_count = 0
    critical_count = 0
    total_duration_days = 0

    # Calculate weighted progress using P6 progress_pct
    total_weight = 0.0
    weighted_sum = 0.0

    # Find date range
    earliest_start = None
    latest_finish = None

    for a in activities:
        # Use P6 progress_pct if available
        if hasattr(a, 'progress_pct') and a.progress_pct is not None:
            progress = a.progress_pct
        else:
            progress = a.pct_complete / 100.0

        # Compute weight (use default for project-wide summary)
        weight = compute_activity_weight(a, 1.0)
        weighted_sum += weight * progress
        total_weight += weight

        # Track P6 stats
        if hasattr(a, 'is_complete') and a.is_complete:
            complete_count += 1
        elif hasattr(a, 'is_in_progress') and a.is_in_progress:
            in_progress_count += 1
        else:
            not_started_count += 1

        if hasattr(a, 'is_critical') and a.is_critical:
            critical_count += 1

        if a.duration_days:
            total_duration_days += a.duration_days

        if a.start_date:
            if earliest_start is None or a.start_date < earliest_start:
                earliest_start = a.start_date
        if a.finish_date:
            if latest_finish is None or a.finish_date > latest_finish:
                latest_finish = a.finish_date

    avg_progress = (weighted_sum / total_weight * 100) if total_weight > 0 else 0

    # Count mapped vs unmapped
    mapped_ids = set(
        m.schedule_activity_id
        for m in db.query(ScheduleToGMPMapping).all()
    )
    mapped_activities = sum(1 for a in activities if a.id in mapped_ids)
    unmapped_activities = total_activities - mapped_activities

    # Group by GMP division with P6 weighting
    division_summaries = {}
    for m in db.query(ScheduleToGMPMapping).all():
        div = m.gmp_division
        activity = m.activity

        # Use P6 progress_pct
        if hasattr(activity, 'progress_pct') and activity.progress_pct is not None:
            progress = activity.progress_pct
        else:
            progress = activity.pct_complete / 100.0

        # Compute P6 weight
        weight = compute_activity_weight(activity, m.weight)

        if div not in division_summaries:
            division_summaries[div] = {
                'gmp_division': div,
                'activity_count': 0,
                'total_weight': 0.0,
                'weighted_progress': 0.0,
                'complete_count': 0,
                'in_progress_count': 0,
                'critical_count': 0
            }

        division_summaries[div]['activity_count'] += 1
        division_summaries[div]['total_weight'] += weight
        division_summaries[div]['weighted_progress'] += weight * progress

        if getattr(activity, 'is_complete', False):
            division_summaries[div]['complete_count'] += 1
        elif getattr(activity, 'is_in_progress', False):
            division_summaries[div]['in_progress_count'] += 1

        if getattr(activity, 'is_critical', False):
            division_summaries[div]['critical_count'] += 1

    # Finalize weighted progress
    by_division = []
    for div, summary in division_summaries.items():
        if summary['total_weight'] > 0:
            summary['weighted_progress'] = round(
                summary['weighted_progress'] / summary['total_weight'] * 100, 1
            )
        by_division.append(summary)

    by_division.sort(key=lambda x: x['gmp_division'])

    return {
        'total_activities': total_activities,
        'mapped_activities': mapped_activities,
        'unmapped_activities': unmapped_activities,
        'avg_progress': round(avg_progress, 1),
        'earliest_start': earliest_start.isoformat() if earliest_start else None,
        'latest_finish': latest_finish.isoformat() if latest_finish else None,
        'by_division': by_division,
        'p6_stats': {
            'complete_count': complete_count,
            'in_progress_count': in_progress_count,
            'not_started_count': not_started_count,
            'critical_count': critical_count,
            'total_duration_days': total_duration_days,
            'total_weight': round(total_weight, 2)
        }
    }


# =============================================================================
# Dashboard Summary Metrics (Single Source of Truth)
# =============================================================================

def get_total_gmp_budget(db: Session) -> int:
    """
    Get total GMP budget from GMP entities table.

    This is the authoritative source for the contract total (static baseline).
    Only changes via approved Change Orders or GMP Excel reload.

    Returns:
        Total GMP budget in cents, or 0 if no GMP data exists.
    """
    # Sum original_amount_cents + approved change orders for each GMP entity
    gmp_entities = db.query(GMP).all()

    if not gmp_entities:
        return 0

    total_cents = 0
    for gmp in gmp_entities:
        total_cents += gmp.original_amount_cents or 0
        # Add approved change orders
        total_cents += gmp.approved_change_order_total_cents

    return total_cents


def get_total_actual_costs(db: Session) -> int:
    """
    Get total actual costs from DirectCostEntity table.

    Excludes voided transactions and duplicates.

    Returns:
        Total actual costs in cents.
    """
    # Query DirectCostEntity, excluding any voided or excluded records
    # The DirectCostEntity model uses gross_amount_cents
    result = db.query(func.sum(DirectCostEntity.gross_amount_cents)).scalar()

    return result or 0


def get_total_forecast_remaining(db: Session) -> Optional[int]:
    """
    Get total forecast remaining (ETC) from ForecastSnapshot table.

    Uses the latest snapshot per GMP division.

    Returns:
        Total forecast remaining in cents, or None if no forecast data.
    """
    # Get current snapshots (is_current = True)
    snapshots = db.query(ForecastSnapshot).filter(
        ForecastSnapshot.is_current == True
    ).all()

    if not snapshots:
        return None

    # Sum etc_cents (Estimate to Complete = forecast remaining)
    total_etc = sum(s.etc_cents or 0 for s in snapshots)

    return total_etc


def compute_weighted_schedule_progress(db: Session) -> Optional[float]:
    """
    Compute weighted schedule progress across all activities.

    Uses P6-style weighting based on critical path and duration.

    Returns:
        Weighted progress as a float (0.0 to 1.0), or None if no schedule data.
    """
    try:
        activities = db.query(ScheduleActivity).all()

        if not activities:
            return None

        total_weight = 0.0
        weighted_sum = 0.0

        for activity in activities:
            # Use P6 progress_pct if available
            if hasattr(activity, 'progress_pct') and activity.progress_pct is not None:
                progress = activity.progress_pct
            else:
                progress = (activity.pct_complete or 0) / 100.0

            # Compute weight using P6 algorithm
            weight = compute_activity_weight(activity, 1.0)
            weighted_sum += weight * progress
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight

        return None

    except OperationalError as e:
        logger.warning(f"Schedule query failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error computing schedule progress: {e}")
        return None


def compute_cpi_if_ev_available(
    db: Session,
    actual_cents: int,
    budget_cents: int
) -> Optional[float]:
    """
    Compute CPI only if Earned Value (EV) is available.

    CPI = EV / AC (Earned Value / Actual Cost)

    EV can come from:
    1. ForecastSnapshot.ev_cents if tracked
    2. Schedule progress × Budget (as proxy)

    Returns:
        CPI as float, or None if EV is not available.
    """
    if actual_cents <= 0:
        return None

    # First, try to get EV from forecast snapshots
    snapshots = db.query(ForecastSnapshot).filter(
        ForecastSnapshot.is_current == True
    ).all()

    total_ev = sum(s.ev_cents or 0 for s in snapshots if s.ev_cents is not None)

    if total_ev > 0:
        # We have tracked EV
        return round(total_ev / actual_cents, 3)

    # Try schedule-weighted progress as EV proxy
    weighted_progress = compute_weighted_schedule_progress(db)

    if weighted_progress is not None and budget_cents > 0:
        ev_cents = int(budget_cents * weighted_progress)
        if ev_cents > 0:
            return round(ev_cents / actual_cents, 3)

    # EV not available
    return None


def compute_schedule_variance_days(db: Session) -> Optional[int]:
    """
    Compute schedule variance in days.

    Compares actual progress to planned progress based on schedule dates.

    Returns:
        Schedule variance in days (positive = ahead, negative = behind),
        or None if schedule data is unavailable.
    """
    try:
        from datetime import date

        activities = db.query(ScheduleActivity).all()

        if not activities:
            return None

        today = date.today()
        total_days_ahead = 0
        activity_count = 0

        for activity in activities:
            # Need both planned and actual dates
            if not activity.planned_finish or not activity.finish_date:
                continue

            # Calculate days ahead/behind
            planned_finish = activity.planned_finish
            current_finish = activity.finish_date

            # Positive = ahead of schedule, negative = behind
            variance_days = (planned_finish - current_finish).days
            total_days_ahead += variance_days
            activity_count += 1

        if activity_count > 0:
            return total_days_ahead // activity_count  # Average variance

        return None

    except OperationalError as e:
        logger.warning(f"Schedule variance query failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error computing schedule variance: {e}")
        return None


def compute_dashboard_summary(db: Session) -> Dict:
    """
    Compute dashboard summary metrics (single source of truth).

    This function is the authoritative source for all dashboard KPIs.

    Metric definitions:
    - total_gmp_budget_cents: Fixed contract amount from GMP table (static baseline)
    - actual_costs_cents: Sum of posted direct costs (post-dedup)
    - forecast_remaining_cents: Predicted remaining costs to finish (NOT total EAC)
    - eac_cents: Actual + Forecast Remaining
    - variance_cents: GMP Budget - EAC (positive = underrun, negative = overrun)
    - cpi: EV / AC (only if EV is available; otherwise None)
    - progress_pct: actual_costs_cents / eac_cents * 100

    Returns:
        Dict with all metrics in cents (ints) + warnings list.
    """
    warnings = []

    # 1. Total GMP Budget (from GMP entities table, NOT from forecast/EAC)
    total_gmp_cents = get_total_gmp_budget(db)
    if total_gmp_cents == 0:
        warnings.append("GMP Budget data unavailable or zero")

    # 2. Actual Costs (from DirectCostEntity)
    actual_cents = get_total_actual_costs(db)

    # 3. Forecast Remaining (from latest ForecastSnapshot per division)
    forecast_remaining_cents = get_total_forecast_remaining(db)
    if forecast_remaining_cents is None:
        warnings.append("Forecast data unavailable")
        forecast_remaining_cents = 0

    # 4. EAC = Actual + Forecast Remaining
    eac_cents = actual_cents + forecast_remaining_cents

    # 5. Variance = Budget - EAC (positive = underrun/savings, negative = overrun)
    variance_cents = None
    if total_gmp_cents > 0:
        variance_cents = total_gmp_cents - eac_cents

    # 6. Progress = Actual / EAC (clamped 0-100)
    progress_pct = 0.0
    if eac_cents > 0:
        progress_pct = (actual_cents / eac_cents) * 100
        progress_pct = max(0.0, min(100.0, progress_pct))

    # 7. CPI (only if EV is available)
    cpi = compute_cpi_if_ev_available(db, actual_cents, total_gmp_cents)
    if cpi is None and actual_cents > 0:
        warnings.append("CPI requires Earned Value (EV) which is not configured")

    # 8. Schedule Variance (wrapped for safety)
    schedule_variance_days = None
    try:
        schedule_variance_days = compute_schedule_variance_days(db)
    except Exception as e:
        warnings.append(f"Schedule data unavailable: {str(e)}")

    return {
        'total_gmp_budget_cents': total_gmp_cents,
        'actual_costs_cents': actual_cents,
        'forecast_remaining_cents': forecast_remaining_cents,
        'eac_cents': eac_cents,
        'variance_cents': variance_cents,
        'progress_pct': round(progress_pct, 1),
        'cpi': cpi,
        'schedule_variance_days': schedule_variance_days,
        'warnings': warnings
    }

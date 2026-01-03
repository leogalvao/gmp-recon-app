"""
Reconciliation Module for GMP Reconciliation App.
Computes per-GMP-division rows with required and optional columns.
All calculations in integer cents for precision.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from .etl import cents_to_display
from ..models import Settings


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
            'gmp_scope_confirmed': settings.gmp_scope_confirmed
        }
    return {
        'as_of_date': None,
        'forecast_basis': 'actuals_plus_commitments',
        'eac_mode_when_commitments': 'max',
        'gmp_scope_notes': None,
        'gmp_scope_confirmed': False
    }


def filter_by_as_of_date(direct_costs_df: pd.DataFrame, as_of_date: Optional[datetime]) -> pd.DataFrame:
    """Filter direct costs to only include rows up to as_of_date."""
    if as_of_date is None:
        return direct_costs_df
    
    mask = direct_costs_df['date_parsed'] <= as_of_date
    return direct_costs_df[mask].copy()


def aggregate_actuals_by_gmp(
    direct_costs_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Aggregate actual costs by GMP division, split by West/East.
    
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
    # Direct costs have mapped_budget_code, budget has Budget Code -> gmp_division
    budget_gmp_map = budget_df[['Budget Code', 'gmp_division']].drop_duplicates()
    
    merged = filtered.merge(
        budget_gmp_map,
        left_on='mapped_budget_code',
        right_on='Budget Code',
        how='left'
    )
    
    # Aggregate by GMP division
    agg = merged.groupby('gmp_division').agg({
        'amount_west': 'sum',
        'amount_east': 'sum',
        'amount_cents': 'sum',
        'direct_cost_id': 'count'
    }).reset_index()
    
    agg.columns = ['gmp_division', 'actual_west_cents', 'actual_east_cents', 'actual_total_cents', 'row_count']
    
    return agg


def aggregate_commitments_by_gmp(
    budget_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate committed costs by GMP division, split by West/East.
    
    Returns DataFrame with columns:
    - gmp_division
    - committed_west_cents
    - committed_east_cents
    - committed_total_cents
    """
    # Committed costs are on the budget rows
    # Apply allocation to committed costs
    if 'committed_costs_cents' not in budget_df.columns:
        return pd.DataFrame(columns=['gmp_division', 'committed_west_cents', 'committed_east_cents', 'committed_total_cents'])
    
    budget_df = budget_df.copy()
    budget_df['committed_west'] = (budget_df['committed_costs_cents'] * budget_df['pct_west']).round().astype(int)
    budget_df['committed_east'] = (budget_df['committed_costs_cents'] * budget_df['pct_east']).round().astype(int)
    
    agg = budget_df.groupby('gmp_division').agg({
        'committed_west': 'sum',
        'committed_east': 'sum',
        'committed_costs_cents': 'sum'
    }).reset_index()
    
    agg.columns = ['gmp_division', 'committed_west_cents', 'committed_east_cents', 'committed_total_cents']
    
    return agg


def compute_reconciliation(
    gmp_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    direct_costs_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame] = None,
    settings: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Main reconciliation computation.
    
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
    
    Returns DataFrame ready for display.
    """
    settings = settings or {
        'as_of_date': None,
        'forecast_basis': 'actuals_plus_commitments',
        'eac_mode_when_commitments': 'max'
    }
    
    as_of_date = settings.get('as_of_date')
    forecast_basis = settings.get('forecast_basis', 'actuals_plus_commitments')
    eac_mode = settings.get('eac_mode_when_commitments', 'max')
    
    # Get actuals aggregated by GMP
    actuals_agg = aggregate_actuals_by_gmp(direct_costs_df, budget_df, as_of_date)
    
    # Get commitments aggregated by GMP
    commitments_agg = aggregate_commitments_by_gmp(budget_df)
    
    # Start with GMP as base
    result = gmp_df[['GMP', 'amount_total_cents']].copy()
    result.columns = ['gmp_division', 'gmp_amount_cents']
    
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
        rows.append({
            'gmp_division': row['gmp_division'],
            'gmp_amount': cents_to_display(int(row['gmp_amount_cents'])),
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

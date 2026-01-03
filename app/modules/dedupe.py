"""
Duplicate Detection Module for GMP Reconciliation App.
Identifies exact and fuzzy duplicates in direct cost entries.
Uses normalized fields and confidence scoring.
"""
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz
from collections import defaultdict
from sqlalchemy.orm import Session

from ..models import Duplicate


# Configuration
FUZZY_VENDOR_THRESHOLD = 90  # out of 100
FUZZY_DESCRIPTION_THRESHOLD = 90
AMOUNT_TOLERANCE_PCT = 0.01  # 1%
DATE_WINDOW_DAYS = 7
AUTO_COLLAPSE_THRESHOLD = 0.98
REVERSAL_WINDOW_DAYS = 14


def find_exact_duplicates(df: pd.DataFrame) -> List[Dict]:
    """
    Find exact duplicates based on:
    (Vendor, normalized_invoice, Amount_cents, Date)
    
    Returns list of duplicate groups with confidence 1.0.
    """
    duplicates = []
    
    # Group by exact match fields
    df_clean = df[df['vendor_clean'].notna() & (df['vendor_clean'] != '')]
    
    grouped = df_clean.groupby(['vendor_clean', 'normalized_invoice', 'amount_cents', 'date_parsed'])
    
    group_id = 0
    for key, group in grouped:
        if len(group) > 1:
            group_id += 1
            for idx, row in group.iterrows():
                duplicates.append({
                    'direct_cost_row_id': row['direct_cost_id'],
                    'group_id': group_id,
                    'method': 'exact',
                    'score': 1.0,
                    'matched_with': [r['direct_cost_id'] for r in group.to_dict('records') if r['direct_cost_id'] != row['direct_cost_id']],
                    'details': {
                        'vendor': row['Vendor'],
                        'invoice': row.get('Invoice #', ''),
                        'amount': row['amount_cents'],
                        'date': str(row['date_parsed'])
                    }
                })
    
    return duplicates


def find_fuzzy_duplicates(df: pd.DataFrame, existing_groups: Dict[int, int]) -> List[Dict]:
    """
    Find fuzzy duplicates based on:
    - Similar vendor names (fuzzy match >= 90)
    - Similar descriptions (fuzzy match >= 90)
    - Amount within ±1%
    - Dates within ±7 days

    Returns list of potential duplicates with confidence scores.

    Optimized using blocking strategy: only compare rows within same
    amount bucket to reduce O(n²) to O(n × bucket_size).
    """
    duplicates = []
    processed_pairs = set()

    # Get the next group ID
    max_group = max(existing_groups.values()) if existing_groups else 0
    group_id = max_group

    # Only consider rows with vendors, excluding already-grouped items
    df_with_vendor = df[
        (df['vendor_clean'].notna()) &
        (df['vendor_clean'] != '') &
        (~df['direct_cost_id'].isin(existing_groups.keys()))
    ].copy()

    if len(df_with_vendor) == 0:
        return duplicates

    # Create amount bucket for blocking (round to nearest $100 for 1% tolerance)
    df_with_vendor['_amount_bucket'] = df_with_vendor['amount_cents'].apply(
        lambda x: round(x / 10000) * 10000 if x != 0 else 0
    )

    # Group by amount bucket for blocking - only compare within buckets
    amount_groups = df_with_vendor.groupby('_amount_bucket')

    for amount_bucket, amount_group in amount_groups:
        if len(amount_group) < 2:
            continue

        rows = amount_group.to_dict('records')

        for i in range(len(rows)):
            row1 = rows[i]

            for j in range(i + 1, len(rows)):
                row2 = rows[j]

                # Skip if already processed
                pair_key = tuple(sorted([row1['direct_cost_id'], row2['direct_cost_id']]))
                if pair_key in processed_pairs:
                    continue

                # Check date proximity first (fast filter)
                date1, date2 = row1['date_parsed'], row2['date_parsed']
                if pd.notna(date1) and pd.notna(date2):
                    date_diff = abs((date1 - date2).days)
                    if date_diff > DATE_WINDOW_DAYS:
                        continue
                    date_score = 1.0 - (date_diff / DATE_WINDOW_DAYS)
                else:
                    date_score = 0.5  # Uncertain

                # Check vendor similarity
                vendor_score = fuzz.ratio(row1['vendor_clean'], row2['vendor_clean'])
                if vendor_score < FUZZY_VENDOR_THRESHOLD:
                    continue

                # Check amount similarity
                amt1, amt2 = row1['amount_cents'], row2['amount_cents']
                if amt1 == 0 and amt2 == 0:
                    amount_score = 1.0
                elif amt1 == 0 or amt2 == 0:
                    continue
                else:
                    diff_pct = abs(amt1 - amt2) / max(abs(amt1), abs(amt2))
                    if diff_pct > AMOUNT_TOLERANCE_PCT:
                        continue
                    amount_score = 1.0 - diff_pct

                # Check description similarity
                desc1 = str(row1.get('Description', '')).lower()
                desc2 = str(row2.get('Description', '')).lower()
                desc_score = fuzz.ratio(desc1, desc2) / 100.0 if desc1 and desc2 else 0.5

                # Compute combined confidence score
                combined_score = (
                    (vendor_score / 100.0) * 0.3 +
                    amount_score * 0.3 +
                    date_score * 0.2 +
                    desc_score * 0.2
                )

                if combined_score >= 0.7:
                    processed_pairs.add(pair_key)
                    group_id += 1

                    for row, row_id in [(row1, row1['direct_cost_id']), (row2, row2['direct_cost_id'])]:
                        duplicates.append({
                            'direct_cost_row_id': row_id,
                            'group_id': group_id,
                            'method': 'fuzzy',
                            'score': combined_score,
                            'matched_with': [row2['direct_cost_id'] if row_id == row1['direct_cost_id'] else row1['direct_cost_id']],
                            'details': {
                                'vendor': row['Vendor'],
                                'invoice': row.get('Invoice #', ''),
                                'amount': row['amount_cents'],
                                'date': str(row['date_parsed']),
                                'vendor_score': vendor_score,
                                'amount_score': amount_score,
                                'date_score': date_score,
                                'desc_score': desc_score
                            }
                        })

    return duplicates


def find_reversals(df: pd.DataFrame) -> List[Dict]:
    """
    Find potential reversal entries.

    Rows with same (Vendor, Invoice #, |Amount|) but opposite sign
    and dates within reversal window are tagged as reversal candidates.

    Optimized using grouping: group by (vendor, invoice, abs_amount)
    then only check within groups for opposite signs.
    """
    reversals = []
    group_id = 10000  # Start high to avoid collision
    processed = set()

    df_with_vendor = df[df['vendor_clean'].notna() & (df['vendor_clean'] != '')].copy()

    if len(df_with_vendor) == 0:
        return reversals

    # Create absolute amount for grouping
    df_with_vendor['_abs_amount'] = df_with_vendor['amount_cents'].abs()

    # Group by vendor, invoice, and absolute amount
    grouped = df_with_vendor.groupby(['vendor_clean', 'normalized_invoice', '_abs_amount'])

    for key, group in grouped:
        if len(group) < 2:
            continue

        # Within this group, look for opposite-sign pairs
        rows = group.to_dict('records')

        for i in range(len(rows)):
            row1 = rows[i]
            if row1['direct_cost_id'] in processed:
                continue

            for j in range(i + 1, len(rows)):
                row2 = rows[j]
                if row2['direct_cost_id'] in processed:
                    continue

                # Check if amounts are opposite
                amt1, amt2 = row1['amount_cents'], row2['amount_cents']
                if amt1 == -amt2 and amt1 != 0:
                    # Check date window
                    date1, date2 = row1['date_parsed'], row2['date_parsed']
                    if pd.notna(date1) and pd.notna(date2):
                        date_diff = abs((date1 - date2).days)
                        if date_diff <= REVERSAL_WINDOW_DAYS:
                            group_id += 1
                            processed.add(row1['direct_cost_id'])
                            processed.add(row2['direct_cost_id'])

                            for row in [row1, row2]:
                                reversals.append({
                                    'direct_cost_row_id': row['direct_cost_id'],
                                    'group_id': group_id,
                                    'method': 'reversal',
                                    'score': 1.0,
                                    'matched_with': [row2['direct_cost_id'] if row['direct_cost_id'] == row1['direct_cost_id'] else row1['direct_cost_id']],
                                    'details': {
                                        'vendor': row['Vendor'],
                                        'invoice': row.get('Invoice #', ''),
                                        'amount': row['amount_cents'],
                                        'date': str(row['date_parsed']),
                                        'reversal_pair': True
                                    }
                                })

    return reversals


def detect_duplicates(df: pd.DataFrame) -> Tuple[List[Dict], Dict[int, int]]:
    """
    Main duplicate detection function.
    
    Returns:
    - List of all detected duplicates with scores
    - Dictionary mapping row_id -> group_id for exact matches
    """
    # Find exact duplicates first
    exact_dups = find_exact_duplicates(df)
    
    # Build mapping of which rows are already in groups
    existing_groups = {d['direct_cost_row_id']: d['group_id'] for d in exact_dups}
    
    # Find fuzzy duplicates (excluding exact matches)
    fuzzy_dups = find_fuzzy_duplicates(df, existing_groups)
    
    # Find reversals
    reversals = find_reversals(df)
    
    all_duplicates = exact_dups + fuzzy_dups + reversals
    
    return all_duplicates, existing_groups


def should_auto_collapse(score: float) -> bool:
    """Determine if a duplicate should be auto-collapsed based on confidence."""
    return score >= AUTO_COLLAPSE_THRESHOLD


def apply_duplicate_exclusions(df: pd.DataFrame, duplicates: List[Dict], 
                                auto_collapse: bool = True) -> pd.DataFrame:
    """
    Apply exclusions to the direct costs DataFrame based on detected duplicates.
    
    For auto-collapse threshold duplicates, mark all but one as excluded.
    For lower confidence, just flag but don't exclude.
    """
    df = df.copy()
    
    if not duplicates:
        return df
    
    # Group duplicates by group_id
    groups = defaultdict(list)
    for dup in duplicates:
        groups[dup['group_id']].append(dup)
    
    for group_id, group_dups in groups.items():
        # Sort by row_id to keep consistent which one to keep
        group_dups.sort(key=lambda x: x['direct_cost_row_id'])
        
        # Check if auto-collapse should apply
        max_score = max(d['score'] for d in group_dups)
        
        if auto_collapse and should_auto_collapse(max_score):
            # Keep first, exclude rest
            for i, dup in enumerate(group_dups):
                if i > 0:
                    mask = df['direct_cost_id'] == dup['direct_cost_row_id']
                    df.loc[mask, 'excluded_from_actuals'] = True
    
    return df


def save_duplicates_to_db(db: Session, duplicates: List[Dict]):
    """
    Save detected duplicates to the database.
    """
    # Clear existing duplicates
    db.query(Duplicate).delete()
    
    for dup in duplicates:
        record = Duplicate(
            direct_cost_row_id=dup['direct_cost_row_id'],
            group_id=dup['group_id'],
            method=dup['method'],
            score=dup['score'],
            resolved=False,
            excluded_from_actuals=should_auto_collapse(dup['score'])
        )
        db.add(record)
    
    db.commit()


def get_duplicates_summary(duplicates: List[Dict]) -> Dict:
    """
    Get summary statistics about detected duplicates.
    """
    if not duplicates:
        return {
            'total_duplicates': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'reversals': 0,
            'auto_collapsed': 0,
            'pending_review': 0,
            'groups': 0
        }
    
    exact = [d for d in duplicates if d['method'] == 'exact']
    fuzzy = [d for d in duplicates if d['method'] == 'fuzzy']
    reversals = [d for d in duplicates if d['method'] == 'reversal']
    
    auto_collapsed = [d for d in duplicates if should_auto_collapse(d['score'])]
    pending = [d for d in duplicates if not should_auto_collapse(d['score'])]
    
    unique_groups = len(set(d['group_id'] for d in duplicates))
    
    return {
        'total_duplicates': len(duplicates),
        'exact_matches': len(exact),
        'fuzzy_matches': len(fuzzy),
        'reversals': len(reversals),
        'auto_collapsed': len(auto_collapsed),
        'pending_review': len(pending),
        'groups': unique_groups
    }


def format_duplicates_for_display(duplicates: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Format duplicates for HTML display with full row details.
    """
    from .etl import cents_to_display
    
    # Build lookup from direct costs
    dc_lookup = df.set_index('direct_cost_id').to_dict('index')
    
    formatted = []
    for dup in duplicates:
        row_id = dup['direct_cost_row_id']
        row_data = dc_lookup.get(row_id, {})
        
        formatted.append({
            'row_id': row_id,
            'group_id': dup['group_id'],
            'method': dup['method'],
            'score': f"{dup['score']:.2%}",
            'score_raw': dup['score'],
            'auto_collapse': should_auto_collapse(dup['score']),
            'cost_code': row_data.get('Cost Code', ''),
            'name': row_data.get('Name', ''),
            'vendor': row_data.get('Vendor', ''),
            'invoice': row_data.get('Invoice #', ''),
            'amount': cents_to_display(row_data.get('amount_cents', 0)),
            'amount_raw': row_data.get('amount_cents', 0),
            'date': str(row_data.get('Date', '')),
            'description': row_data.get('Description', ''),
            'matched_with': dup.get('matched_with', []),
            'details': dup.get('details', {})
        })
    
    # Sort by group_id, then by row_id
    formatted.sort(key=lambda x: (x['group_id'], x['row_id']))
    
    return formatted

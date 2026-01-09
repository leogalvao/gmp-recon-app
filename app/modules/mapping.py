"""
Mapping Module for GMP Reconciliation App.
Handles two-layer mapping: Budget → GMP and Direct Cost → Budget.
Uses code prefix matching with fuzzy text fallback.

Configuration is loaded from gmp_mapping_config.yaml via app.config.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session
import json
from datetime import datetime, timezone

from ..models import BudgetToGMP, DirectToBudget, Allocation, MappingAudit
from ..config import get_config
from .etl import allocate_east_west


# =============================================================================
# Configuration Accessors
# =============================================================================

def _get_division_lookup() -> Dict[str, str]:
    """Build division key to GMP name lookup from config."""
    config = get_config()
    return {key: div.get('name', '') for key, div in config.gmp_divisions.items()}


def _get_fuzzy_thresholds(context: str) -> Dict[str, int]:
    """Get fuzzy matching thresholds for a context."""
    return get_config().get_fuzzy_thresholds(context)


def _get_default_allocation() -> Tuple[float, float]:
    """Get default West/East allocation from config."""
    config = get_config()
    default = config.default_allocation
    return default.get('west', 0.5), default.get('east', 0.5)


def fuzzy_match_gmp(text: str, gmp_names: List[str], threshold: Optional[int] = None) -> Tuple[Optional[str], float]:
    """
    Fuzzy match a text string to GMP division names.
    Returns (best_match, confidence_score) or (None, 0) if no match above threshold.

    Args:
        text: Text to match
        gmp_names: List of GMP division names to match against
        threshold: Minimum score (0-100) to accept. If None, uses config default.
    """
    if not text or not gmp_names:
        return None, 0.0

    # Use config threshold if not specified
    if threshold is None:
        thresholds = _get_fuzzy_thresholds('budget_to_gmp')
        threshold = thresholds.get('min_confidence', 85)

    result = process.extractOne(text, gmp_names, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0], result[1] / 100.0
    return None, 0.0


def map_budget_to_gmp(budget_df: pd.DataFrame, gmp_df: pd.DataFrame,
                       db: Optional[Session] = None) -> pd.DataFrame:
    """
    Map Budget rows to GMP divisions.

    Algorithm:
    1. Check for existing mapping in database
    2. Extract division key from Cost Code Tier 2 (prefix like "1-010" → "1")
    3. Map division numbers to GMP rows via config gmp_divisions
    4. If missing, fuzzy-match Budget Code Description to GMP names

    Returns budget_df with added columns: gmp_division, mapping_confidence, mapping_method
    """
    gmp_names = gmp_df['GMP'].tolist()
    gmp_lookup = {name.strip().lower(): name for name in gmp_names}

    # Get division lookup from config
    division_to_gmp = _get_division_lookup()

    # Get fuzzy thresholds from config
    tier2_thresholds = _get_fuzzy_thresholds('tier2_to_gmp')

    # Load existing mappings from database
    db_mappings = {}
    if db:
        existing = db.query(BudgetToGMP).all()
        db_mappings = {m.budget_code: (m.gmp_division, m.confidence) for m in existing}

    results = []
    for idx, row in budget_df.iterrows():
        budget_code = row.get('Budget Code', '')
        tier2 = row.get('Cost Code Tier 2', '')
        description = row.get('Budget Code Description', '')
        division_key = row.get('division_key', '')

        gmp_division = None
        confidence = 0.0
        method = 'unmapped'

        # Priority 1: Check database for existing mapping
        if budget_code and budget_code in db_mappings:
            gmp_division, confidence = db_mappings[budget_code]
            method = 'database'

        # Priority 2: Use division key lookup from config
        elif division_key and division_key in division_to_gmp:
            candidate = division_to_gmp[division_key]
            # Verify it exists in GMP data
            if candidate.strip().lower() in gmp_lookup:
                gmp_division = gmp_lookup[candidate.strip().lower()]
                confidence = 0.95
                method = 'division_key'

        # Priority 3: Fuzzy match on description
        if gmp_division is None and description and isinstance(description, str):
            matched, score = fuzzy_match_gmp(description, gmp_names)
            if matched:
                gmp_division = matched
                confidence = score
                method = 'fuzzy_description'

        # Priority 4: Fuzzy match on tier 2 name (uses tier2_to_gmp thresholds)
        if gmp_division is None and tier2 and isinstance(tier2, str):
            # Extract name part after the code
            name_part = tier2.split(' - ', 1)[1] if ' - ' in tier2 else tier2
            tier2_threshold = tier2_thresholds.get('min_confidence', 80)
            matched, score = fuzzy_match_gmp(name_part, gmp_names, threshold=tier2_threshold)
            if matched:
                gmp_division = matched
                confidence = score
                method = 'fuzzy_tier2'

        results.append({
            'budget_id': row.get('budget_id', idx),
            'gmp_division': gmp_division,
            'mapping_confidence': confidence,
            'mapping_method': method
        })

    mapping_df = pd.DataFrame(results)
    budget_df = budget_df.merge(mapping_df, left_on='budget_id', right_on='budget_id', how='left')

    return budget_df


def map_direct_to_budget(direct_df: pd.DataFrame, budget_df: pd.DataFrame,
                          db: Optional[Session] = None) -> pd.DataFrame:
    """
    Map Direct Cost rows to Budget codes.

    Algorithm:
    1. Check for existing mapping in database
    2. Join by Cost Code to Budget rows sharing the same base prefix
    3. If multiple candidates, prefer matching Cost Type (Labor/Material)
    4. If still ambiguous, fuzzy-match Name to Budget Code Description

    Returns direct_df with added columns: budget_code, mapping_confidence, mapping_method
    """
    # Get fuzzy thresholds from config
    direct_thresholds = _get_fuzzy_thresholds('direct_to_budget')
    min_fuzzy_threshold = direct_thresholds.get('min_confidence', 60)

    # Build lookup of budget codes by base code
    budget_lookup = {}
    for idx, row in budget_df.iterrows():
        base_code = row.get('base_code', '')
        if base_code:
            if base_code not in budget_lookup:
                budget_lookup[base_code] = []
            budget_lookup[base_code].append({
                'budget_code': row.get('Budget Code', ''),
                'description': row.get('Budget Code Description', ''),
                'cost_type': row.get('Cost Type', '')
            })

    # Load existing mappings from database
    db_mappings = {}
    if db:
        existing = db.query(DirectToBudget).all()
        for m in existing:
            key = (m.cost_code, m.name)
            db_mappings[key] = (m.budget_code, m.confidence)

    results = []
    for idx, row in direct_df.iterrows():
        cost_code = row.get('Cost Code', '')
        base_code = row.get('base_code', '')
        name = row.get('Name', '')
        dc_type = row.get('Type', '')

        budget_code = None
        confidence = 0.0
        method = 'unmapped'

        # Priority 1: Check database for existing mapping
        key = (cost_code, name)
        if key in db_mappings:
            budget_code, confidence = db_mappings[key]
            method = 'database'

        # Priority 2: Direct match by base code
        elif base_code and base_code in budget_lookup:
            candidates = budget_lookup[base_code]

            if len(candidates) == 1:
                budget_code = candidates[0]['budget_code']
                confidence = 0.9
                method = 'base_code_exact'
            else:
                # Try to match by cost type
                type_map = {
                    'Labor': ['L', 'Labor', 'LB'],
                    'Material': ['M', 'Material'],
                    'Other': ['O', 'Other'],
                    'Subcontract': ['S', 'Subcontract', 'SC']
                }

                matched_by_type = None
                for cand in candidates:
                    cand_type_code = cand['cost_type'].split(' - ')[0] if ' - ' in cand['cost_type'] else cand['cost_type']

                    # Check if direct cost type matches budget cost type
                    for type_name, codes in type_map.items():
                        if dc_type in codes or dc_type == type_name:
                            if cand_type_code in codes or cand_type_code.startswith(type_name[0]):
                                matched_by_type = cand
                                break
                    if matched_by_type:
                        break

                if matched_by_type:
                    budget_code = matched_by_type['budget_code']
                    confidence = 0.85
                    method = 'base_code_type'
                else:
                    # Fuzzy match on description (uses config threshold)
                    descriptions = [c['description'] for c in candidates]
                    if name and descriptions:
                        result = process.extractOne(name, descriptions, scorer=fuzz.token_sort_ratio)
                        if result and result[1] >= min_fuzzy_threshold:
                            best_idx = descriptions.index(result[0])
                            budget_code = candidates[best_idx]['budget_code']
                            confidence = result[1] / 100.0
                            method = 'fuzzy_match'
                        else:
                            # Default to first candidate if nothing else matches
                            budget_code = candidates[0]['budget_code']
                            confidence = 0.5
                            method = 'base_code_default'

        results.append({
            'direct_cost_id': row.get('direct_cost_id', idx),
            'mapped_budget_code': budget_code,
            'dc_mapping_confidence': confidence,
            'dc_mapping_method': method
        })

    mapping_df = pd.DataFrame(results)
    direct_df = direct_df.merge(mapping_df, left_on='direct_cost_id', right_on='direct_cost_id', how='left')

    return direct_df


def build_allocation_cache(db: Optional[Session] = None,
                           allocations_df: Optional[pd.DataFrame] = None) -> Dict[str, Tuple[float, float, bool]]:
    """
    Build an in-memory cache of all allocations for O(1) lookup.

    Args:
        db: Database session (allocations from Allocation table)
        allocations_df: DataFrame with allocations from CSV

    Returns:
        Dict mapping code -> (pct_west, pct_east, confirmed)
    """
    cache = {}

    # Load from DataFrame first (lower priority)
    if allocations_df is not None and not allocations_df.empty:
        for _, row in allocations_df.iterrows():
            code = str(row.get('code', ''))
            if code:
                cache[code] = (
                    float(row.get('pct_west', 0.5)),
                    float(row.get('pct_east', 0.5)),
                    bool(row.get('confirmed', True))
                )

    # Load from database (higher priority, overwrites CSV)
    if db:
        allocations = db.query(Allocation).all()
        for alloc in allocations:
            cache[alloc.code] = (alloc.pct_west, alloc.pct_east, alloc.confirmed)

    return cache


def get_allocation(code: str, allocations_df: pd.DataFrame,
                   db: Optional[Session] = None,
                   cache: Optional[Dict[str, Tuple[float, float, bool]]] = None) -> Tuple[float, float, bool]:
    """
    Get West/East allocation percentages for a code.
    Returns (pct_west, pct_east, confirmed).
    Default allocation is loaded from config if no allocation found.

    Args:
        code: Cost code to look up
        allocations_df: DataFrame with allocations from CSV
        db: Database session (for individual lookups if no cache)
        cache: Pre-built allocation cache for O(1) lookup
    """
    # Use cache if provided (O(1) lookup)
    if cache is not None and code in cache:
        return cache[code]

    # Fallback to database query (avoid if using cache)
    if db:
        alloc = db.query(Allocation).filter(Allocation.code == code).first()
        if alloc:
            return alloc.pct_west, alloc.pct_east, alloc.confirmed

    # Check allocations DataFrame
    if not allocations_df.empty:
        match = allocations_df[allocations_df['code'] == code]
        if not match.empty:
            row = match.iloc[0]
            return row['pct_west'], row['pct_east'], row.get('confirmed', True)

    # Default allocation from config (unconfirmed)
    default_west, default_east = _get_default_allocation()
    return default_west, default_east, False


def apply_allocations(df: pd.DataFrame, amount_col: str, code_col: str,
                      allocations_df: pd.DataFrame, db: Optional[Session] = None) -> pd.DataFrame:
    """
    Apply West/East allocations to a DataFrame.
    Adds columns: amount_west, amount_east, allocation_confirmed.

    Uses pre-built cache to avoid N+1 query pattern.
    """
    # Build cache once before iterating (O(n) -> O(1) per row)
    cache = build_allocation_cache(db, allocations_df)
    default_west, default_east = _get_default_allocation()

    results = []
    for idx, row in df.iterrows():
        code = row.get(code_col, '')
        amount = row.get(amount_col, 0)

        # Use cache for O(1) lookup
        if code in cache:
            pct_west, pct_east, confirmed = cache[code]
        else:
            pct_west, pct_east, confirmed = default_west, default_east, False

        # Use Largest Remainder Method for penny-perfect allocation
        # This ensures West + East = amount exactly with fair rounding
        amount_east, amount_west = allocate_east_west(int(amount), pct_east, pct_west)

        results.append({
            'row_idx': idx,
            'amount_west': amount_west,
            'amount_east': amount_east,
            'pct_west': pct_west,
            'pct_east': pct_east,
            'allocation_confirmed': confirmed
        })

    alloc_df = pd.DataFrame(results)
    df = df.reset_index(drop=True)
    df = pd.concat([df, alloc_df[['amount_west', 'amount_east', 'pct_west', 'pct_east', 'allocation_confirmed']]], axis=1)

    return df


def save_mapping(db: Session, table: str, data: Dict, user: str = 'system') -> Dict:
    """
    Save a mapping to the database with audit trail.

    Returns:
        Dict with 'action' ('created' or 'updated') and 'id' (record ID)
    """
    timestamp = datetime.now(timezone.utc)
    action_taken = 'created'
    record_id = None

    if table == 'budget_to_gmp':
        existing = db.query(BudgetToGMP).filter(BudgetToGMP.budget_code == data['budget_code']).first()
        old_value = None
        if existing:
            action_taken = 'updated'
            old_value = json.dumps({
                'gmp_division': existing.gmp_division,
                'side': existing.side,
                'confidence': existing.confidence
            })
            # Update all relevant fields
            existing.gmp_division = data['gmp_division']
            existing.confidence = data.get('confidence', 1.0)
            existing.side = data.get('side', existing.side)  # Preserve existing if not provided
            if 'cost_code_tier2' in data:
                existing.cost_code_tier2 = data['cost_code_tier2']
            existing.updated_at = timestamp
            record_id = existing.id
        else:
            # Ensure defaults for new records
            if 'side' not in data:
                data['side'] = 'BOTH'
            new_record = BudgetToGMP(**data)
            db.add(new_record)
            db.flush()
            record_id = new_record.id

        audit = MappingAudit(
            table_name=table,
            record_id=record_id,
            action='update' if existing else 'create',
            old_value=old_value,
            new_value=json.dumps(data),
            user=user,
            timestamp=timestamp
        )
        db.add(audit)

    elif table == 'direct_to_budget':
        existing = db.query(DirectToBudget).filter(
            DirectToBudget.cost_code == data['cost_code'],
            DirectToBudget.name == data['name']
        ).first()
        old_value = None
        if existing:
            action_taken = 'updated'
            old_value = json.dumps({
                'budget_code': existing.budget_code,
                'side': existing.side,
                'method': existing.method,
                'vendor_normalized': existing.vendor_normalized,
                'confidence': existing.confidence
            })
            # Update all relevant fields
            existing.budget_code = data['budget_code']
            existing.confidence = data.get('confidence', 1.0)
            existing.side = data.get('side', existing.side)  # Preserve existing if not provided
            if 'method' in data:
                existing.method = data['method']
            if 'vendor_normalized' in data:
                existing.vendor_normalized = data['vendor_normalized']
            existing.updated_at = timestamp
            record_id = existing.id
        else:
            # Ensure defaults for new records
            if 'side' not in data:
                data['side'] = 'BOTH'
            if 'method' not in data:
                data['method'] = 'manual'
            new_record = DirectToBudget(**data)
            db.add(new_record)
            db.flush()
            record_id = new_record.id

        audit = MappingAudit(
            table_name=table,
            record_id=record_id,
            action='update' if existing else 'create',
            old_value=old_value,
            new_value=json.dumps(data),
            user=user,
            timestamp=timestamp
        )
        db.add(audit)

    elif table == 'allocations':
        existing = db.query(Allocation).filter(Allocation.code == data['code']).first()
        old_value = None
        if existing:
            action_taken = 'updated'
            old_value = json.dumps({
                'region': existing.region,
                'pct_west': existing.pct_west,
                'pct_east': existing.pct_east,
                'confirmed': existing.confirmed
            })
            existing.region = data.get('region', 'Both')
            existing.pct_west = data['pct_west']
            existing.pct_east = data['pct_east']
            existing.confirmed = data.get('confirmed', True)
            existing.updated_at = timestamp
            record_id = existing.id
        else:
            new_record = Allocation(**data)
            db.add(new_record)
            db.flush()
            record_id = new_record.id

        audit = MappingAudit(
            table_name=table,
            record_id=record_id,
            action='update' if existing else 'create',
            old_value=old_value,
            new_value=json.dumps(data),
            user=user,
            timestamp=timestamp
        )
        db.add(audit)

    db.commit()
    return {'action': action_taken, 'id': record_id}


def get_mapping_stats(budget_df: pd.DataFrame, direct_df: pd.DataFrame) -> Dict:
    """
    Compute mapping statistics for the summary panel.
    """
    total_budget = len(budget_df)
    mapped_budget = len(budget_df[budget_df['gmp_division'].notna()])
    unmapped_budget = total_budget - mapped_budget
    
    total_direct = len(direct_df)
    mapped_direct = len(direct_df[direct_df['mapped_budget_code'].notna()])
    unmapped_direct = total_direct - mapped_direct
    
    # Unconfirmed allocations
    if 'allocation_confirmed' in budget_df.columns:
        unconfirmed_alloc = len(budget_df[budget_df['allocation_confirmed'] == False])
    else:
        unconfirmed_alloc = 0
    
    return {
        'total_budget_rows': total_budget,
        'mapped_budget_to_gmp': mapped_budget,
        'unmapped_budget_to_gmp': unmapped_budget,
        'budget_mapping_pct': mapped_budget / total_budget * 100 if total_budget > 0 else 0,
        'total_direct_rows': total_direct,
        'mapped_direct_to_budget': mapped_direct,
        'unmapped_direct_to_budget': unmapped_direct,
        'direct_mapping_pct': mapped_direct / total_direct * 100 if total_direct > 0 else 0,
        'unconfirmed_allocations': unconfirmed_alloc
    }

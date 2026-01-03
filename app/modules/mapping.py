"""
Mapping Module for GMP Reconciliation App.
Handles two-layer mapping: Budget → GMP and Direct Cost → Budget.
Uses code prefix matching with fuzzy text fallback.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session
import json
from datetime import datetime

from ..models import BudgetToGMP, DirectToBudget, Allocation, MappingAudit


# Default mapping from division keys to GMP names
# This provides a seed for automatic mapping
DEFAULT_DIVISION_TO_GMP = {
    '1': 'General Conditions',
    '2': 'Site Demolition(No Abatement)',
    '3': 'Sitework',
    '4': 'Concrete',
    '5': 'Masonry',
    '6': 'Structural Steel',
    '7': 'Rough Carpentry, Drywall, Ceilings',
    '8': 'Architectural Millwork',
    '9': 'Waterproofing',
    '10': 'Roofing',
    '11': 'Doors, Frames, & Hardware',
    '12': 'Overhead Door',
    '13': 'Aluminum & Glass',
    '14': 'Ceramic Tile',
    '15': 'Flooring',
    '16': 'Painting',
    '17': 'Signage',
    '18': 'Specialties',
    '19': 'Food Service Equipment',
    '20': 'Furniture, Fixtures, & Equipment',
    '21': 'Window Treatments',
    '22': 'Elevators ',  # Note: trailing space in GMP data
    '23': 'Fire Protection',
    '24': 'Plumbing & H.V.A.C',
    '25': 'Geotherm work',
    '26': 'Electrical & Fire Alarm',
    '27': 'Low Voltage',
    '28': 'Landscaping',
    '29': 'Design Fees',  # Could also be General Conditions
    '30': 'Preconstruction Fee',
    '31': 'Design-Build Fee',
}


def fuzzy_match_gmp(text: str, gmp_names: List[str], threshold: int = 85) -> Tuple[Optional[str], float]:
    """
    Fuzzy match a text string to GMP division names.
    Returns (best_match, confidence_score) or (None, 0) if no match above threshold.
    """
    if not text or not gmp_names:
        return None, 0.0
    
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
    3. Map division numbers to GMP rows via DEFAULT_DIVISION_TO_GMP
    4. If missing, fuzzy-match Budget Code Description to GMP names
    
    Returns budget_df with added columns: gmp_division, mapping_confidence, mapping_method
    """
    gmp_names = gmp_df['GMP'].tolist()
    gmp_lookup = {name.strip().lower(): name for name in gmp_names}
    
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
        
        # Priority 2: Use division key lookup
        elif division_key and division_key in DEFAULT_DIVISION_TO_GMP:
            candidate = DEFAULT_DIVISION_TO_GMP[division_key]
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
        
        # Priority 4: Fuzzy match on tier 2 name
        if gmp_division is None and tier2 and isinstance(tier2, str):
            # Extract name part after the code
            name_part = tier2.split(' - ', 1)[1] if ' - ' in tier2 else tier2
            matched, score = fuzzy_match_gmp(name_part, gmp_names, threshold=80)
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
                    # Fuzzy match on description
                    descriptions = [c['description'] for c in candidates]
                    if name and descriptions:
                        result = process.extractOne(name, descriptions, scorer=fuzz.token_sort_ratio)
                        if result and result[1] >= 70:
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


def get_allocation(code: str, allocations_df: pd.DataFrame, 
                   db: Optional[Session] = None) -> Tuple[float, float, bool]:
    """
    Get West/East allocation percentages for a code.
    Returns (pct_west, pct_east, confirmed).
    Default is 50/50 if no allocation found.
    """
    # Check database first
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
    
    # Default 50/50 unconfirmed
    return 0.5, 0.5, False


def apply_allocations(df: pd.DataFrame, amount_col: str, code_col: str,
                      allocations_df: pd.DataFrame, db: Optional[Session] = None) -> pd.DataFrame:
    """
    Apply West/East allocations to a DataFrame.
    Adds columns: amount_west, amount_east, allocation_confirmed.
    """
    results = []
    for idx, row in df.iterrows():
        code = row.get(code_col, '')
        amount = row.get(amount_col, 0)
        
        pct_west, pct_east, confirmed = get_allocation(code, allocations_df, db)
        
        # Calculate West first, then derive East to ensure West + East = amount exactly
        # This prevents rounding accumulation errors
        amount_west = int(round(amount * pct_west))
        amount_east = int(amount) - amount_west  # Remainder goes to East

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


def save_mapping(db: Session, table: str, data: Dict, user: str = 'system'):
    """
    Save a mapping to the database with audit trail.
    """
    timestamp = datetime.utcnow()
    
    if table == 'budget_to_gmp':
        existing = db.query(BudgetToGMP).filter(BudgetToGMP.budget_code == data['budget_code']).first()
        old_value = None
        if existing:
            old_value = json.dumps({'gmp_division': existing.gmp_division, 'confidence': existing.confidence})
            existing.gmp_division = data['gmp_division']
            existing.confidence = data.get('confidence', 1.0)
            existing.updated_at = timestamp
            record_id = existing.id
        else:
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
            old_value = json.dumps({'budget_code': existing.budget_code, 'confidence': existing.confidence})
            existing.budget_code = data['budget_code']
            existing.confidence = data.get('confidence', 1.0)
            existing.updated_at = timestamp
            record_id = existing.id
        else:
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

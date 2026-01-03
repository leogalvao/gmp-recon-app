"""
Suggestion Engine for Direct Cost → Budget Mapping.

Implements the match scoring algorithm with:
- Code prefix matching (40% weight)
- Cost type matching (20% weight)
- Fuzzy text similarity (30% weight)
- Historical pattern boost (10% weight)

Confidence bands:
- High: ≥ 0.85 (auto-suggest, green highlight)
- Medium: 0.60 – 0.84 (show suggestion, yellow highlight)
- Low: < 0.60 (no suggestion, manual required)
"""
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from ..models import (
    MappingFeedback, BudgetMatchStats, SuggestionCache, DirectToBudget
)


# =============================================================================
# Configuration
# =============================================================================

# Scoring weights (must sum to 1.0)
WEIGHT_CODE_MATCH = 0.40
WEIGHT_TYPE_MATCH = 0.20
WEIGHT_TEXT_SIM = 0.30
WEIGHT_HISTORICAL = 0.10

# Confidence thresholds
THRESHOLD_HIGH = 0.85
THRESHOLD_MEDIUM = 0.60
THRESHOLD_MINIMUM = 0.40  # Below this, don't even consider as candidate

# Cost type compatibility matrix
# Full match = 1.0, Partial match = 0.5, No match = 0.0
TYPE_COMPATIBILITY = {
    ('L', 'L'): 1.0,  # Labor
    ('M', 'M'): 1.0,  # Material
    ('S', 'S'): 1.0,  # Subcontract
    ('O', 'O'): 1.0,  # Other
    ('L', 'S'): 0.5,  # Labor ↔ Subcontract (often interchangeable)
    ('S', 'L'): 0.5,
    ('LB', 'L'): 1.0,  # Alternate codes
    ('L', 'LB'): 1.0,
    ('SC', 'S'): 1.0,
    ('S', 'SC'): 1.0,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScoredMatch:
    """A budget code match with its score breakdown."""
    budget_code: str
    description: str
    total_score: float
    code_match_score: float
    type_match_score: float
    text_sim_score: float
    historical_score: float
    confidence_band: str  # 'high', 'medium', 'low'
    historical_match_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'budget_code': self.budget_code,
            'description': self.description,
            'score': round(self.total_score * 100),  # Convert to percentage
            'total_score': round(self.total_score, 3),
            'breakdown': {
                'code_match': round(self.code_match_score, 3),
                'type_match': round(self.type_match_score, 3),
                'text_sim': round(self.text_sim_score, 3),
                'historical': round(self.historical_score, 3),
            },
            'confidence_band': self.confidence_band,
            'historical_match_count': self.historical_match_count,
        }


@dataclass
class DirectCostRow:
    """Normalized direct cost row for scoring."""
    id: int
    cost_code: str
    base_code: str
    name: str
    cost_type: str
    vendor: str
    vendor_normalized: str
    amount_cents: int


@dataclass
class BudgetRow:
    """Normalized budget row for scoring."""
    budget_code: str
    base_code: str
    description: str
    cost_type: str
    cost_type_code: str  # Just the letter code (L, M, S, O)


# =============================================================================
# Normalization Functions
# =============================================================================

def extract_base_code(code: str) -> str:
    """
    Extract base code prefix from cost/budget code.
    Examples:
        '4-010-200' → '4-010'
        '4-010' → '4-010'
        '4' → '4'
    """
    if not code or not isinstance(code, str):
        return ''

    # Match pattern: digit(s) optionally followed by -digit(s)
    match = re.match(r'^(\d+(?:-\d+)?)', code.strip())
    return match.group(1) if match else ''


def normalize_vendor(vendor: str) -> str:
    """
    Normalize vendor name for consistent matching.
    Removes common suffixes, lowercases, strips whitespace.
    """
    if not vendor or not isinstance(vendor, str):
        return ''

    normalized = vendor.lower().strip()

    # Remove common business suffixes
    suffixes = [
        ' incorporated', ' inc', ' llc', ' corp', ' corporation',
        ' co', ' ltd', ' limited', ' company', ' enterprises',
        ' services', ' group', '.', ','
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def extract_name_prefix(name: str, max_length: int = 20) -> str:
    """Extract normalized prefix from name for pattern matching."""
    if not name or not isinstance(name, str):
        return ''
    return name[:max_length].lower().strip()


def extract_cost_type_code(cost_type: str) -> str:
    """
    Extract single-letter cost type code.
    Examples:
        'L - Labor' → 'L'
        'Labor' → 'L'
        'M' → 'M'
    """
    if not cost_type or not isinstance(cost_type, str):
        return ''

    cost_type = cost_type.strip().upper()

    # If already a single letter
    if len(cost_type) == 1:
        return cost_type

    # If format is "X - Description"
    if ' - ' in cost_type:
        return cost_type.split(' - ')[0].strip()

    # Try to match full names
    type_map = {
        'LABOR': 'L',
        'MATERIAL': 'M',
        'MATERIALS': 'M',
        'SUBCONTRACT': 'S',
        'SUB': 'S',
        'OTHER': 'O',
        'EQUIPMENT': 'O',
    }

    for name, code in type_map.items():
        if cost_type.startswith(name):
            return code

    return cost_type[0] if cost_type else ''


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    if not text or not isinstance(text, str):
        return ''

    # Lowercase
    text = text.lower()

    # Remove special characters except spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =============================================================================
# Scoring Functions
# =============================================================================

def compute_code_match(dc_base_code: str, budget_base_code: str) -> float:
    """
    Compute code match score.
    Returns 1.0 for exact match, 0.0 otherwise.
    """
    if not dc_base_code or not budget_base_code:
        return 0.0

    return 1.0 if dc_base_code == budget_base_code else 0.0


def compute_type_match(dc_type: str, budget_type: str) -> float:
    """
    Compute cost type match score using compatibility matrix.
    Returns 1.0 for exact match, 0.5 for compatible types, 0.0 otherwise.
    """
    dc_code = extract_cost_type_code(dc_type)
    budget_code = extract_cost_type_code(budget_type)

    if not dc_code or not budget_code:
        return 0.0

    # Check compatibility matrix
    key = (dc_code, budget_code)
    if key in TYPE_COMPATIBILITY:
        return TYPE_COMPATIBILITY[key]

    # Fallback: exact match only
    return 1.0 if dc_code == budget_code else 0.0


def compute_text_similarity(dc_name: str, budget_description: str) -> float:
    """
    Compute fuzzy text similarity score.
    Uses token_set_ratio for best matching of partial/reordered tokens.
    Returns score from 0.0 to 1.0.
    """
    name_norm = normalize_text(dc_name)
    desc_norm = normalize_text(budget_description)

    if not name_norm or not desc_norm:
        return 0.0

    # token_set_ratio handles partial matches and word reordering well
    score = fuzz.token_set_ratio(name_norm, desc_norm)

    return score / 100.0


def compute_historical_boost(
    vendor_normalized: str,
    name_prefix: str,
    budget_code: str,
    history: Dict[Tuple[str, str], str]
) -> float:
    """
    Compute historical pattern boost.
    Returns 1.0 if this (vendor, name_prefix) was previously mapped to this budget.
    """
    if not vendor_normalized and not name_prefix:
        return 0.0

    key = (vendor_normalized, name_prefix)
    historical_budget = history.get(key)

    return 1.0 if historical_budget == budget_code else 0.0


def compute_match_score(
    dc: DirectCostRow,
    budget: BudgetRow,
    history: Dict[Tuple[str, str], str]
) -> ScoredMatch:
    """
    Compute composite match score for a direct cost → budget pair.

    Score formula:
        score = (0.40 × code_match) + (0.20 × type_match) +
                (0.30 × text_sim) + (0.10 × historical_boost)
    """
    # Component scores
    code_match = compute_code_match(dc.base_code, budget.base_code)
    type_match = compute_type_match(dc.cost_type, budget.cost_type_code)
    text_sim = compute_text_similarity(dc.name, budget.description)
    historical = compute_historical_boost(
        dc.vendor_normalized,
        extract_name_prefix(dc.name),
        budget.budget_code,
        history
    )

    # Weighted sum
    total_score = (
        WEIGHT_CODE_MATCH * code_match +
        WEIGHT_TYPE_MATCH * type_match +
        WEIGHT_TEXT_SIM * text_sim +
        WEIGHT_HISTORICAL * historical
    )

    # Determine confidence band
    if total_score >= THRESHOLD_HIGH:
        confidence_band = 'high'
    elif total_score >= THRESHOLD_MEDIUM:
        confidence_band = 'medium'
    else:
        confidence_band = 'low'

    return ScoredMatch(
        budget_code=budget.budget_code,
        description=budget.description,
        total_score=total_score,
        code_match_score=code_match,
        type_match_score=type_match,
        text_sim_score=text_sim,
        historical_score=historical,
        confidence_band=confidence_band,
    )


# =============================================================================
# Historical Pattern Loading
# =============================================================================

def load_historical_patterns(db: Session) -> Dict[Tuple[str, str], str]:
    """
    Load historical mapping patterns from mapping_feedback table.
    Returns dict: (vendor_normalized, name_prefix) → budget_code

    For patterns with multiple budget codes, uses the most recent non-override.
    """
    patterns = {}

    # Query patterns, prioritizing non-overrides and recent entries
    feedbacks = db.query(MappingFeedback).order_by(
        MappingFeedback.was_override.asc(),  # Non-overrides first
        MappingFeedback.created_at.desc()     # Most recent first
    ).all()

    for fb in feedbacks:
        key = (fb.vendor_normalized, fb.name_prefix)
        if key not in patterns:  # First match wins (non-override, most recent)
            patterns[key] = fb.budget_code

    return patterns


def load_budget_match_counts(db: Session) -> Dict[str, int]:
    """
    Load historical match counts per budget code for tie-breaking.
    Returns dict: budget_code → total_matches
    """
    stats = db.query(BudgetMatchStats).all()
    return {s.budget_code: s.total_matches for s in stats}


def load_budget_trust_scores(db: Session) -> Dict[str, float]:
    """
    Load trust scores per budget code.
    Trust score decays when users override suggestions.
    """
    stats = db.query(BudgetMatchStats).all()
    return {s.budget_code: s.trust_score for s in stats}


# =============================================================================
# Main Ranking Function
# =============================================================================

def rank_suggestions(
    dc: DirectCostRow,
    budget_rows: List[BudgetRow],
    history: Dict[Tuple[str, str], str],
    match_counts: Dict[str, int],
    top_k: int = 3
) -> List[ScoredMatch]:
    """
    Rank all budget rows for a direct cost and return top-k suggestions.

    Tie-breaking rules (when scores are equal):
    1. Prefer budget code with more historical matches
    2. Alphabetical by budget code (deterministic fallback)
    """
    candidates = []

    for budget in budget_rows:
        scored = compute_match_score(dc, budget, history)

        # Only consider candidates above minimum threshold
        if scored.total_score >= THRESHOLD_MINIMUM:
            scored.historical_match_count = match_counts.get(budget.budget_code, 0)
            candidates.append(scored)

    # Sort by: score DESC, historical_count DESC, budget_code ASC
    candidates.sort(key=lambda x: (
        -x.total_score,
        -x.historical_match_count,
        x.budget_code
    ))

    return candidates[:top_k]


# =============================================================================
# DataFrame Conversion Helpers
# =============================================================================

def df_to_direct_cost_rows(df: pd.DataFrame) -> List[DirectCostRow]:
    """Convert DataFrame to list of DirectCostRow objects."""
    rows = []

    for idx, row in df.iterrows():
        cost_code = str(row.get('Cost Code', '') or '')
        vendor = str(row.get('Vendor', '') or '')

        rows.append(DirectCostRow(
            id=row.get('direct_cost_id', idx),
            cost_code=cost_code,
            base_code=extract_base_code(cost_code),
            name=str(row.get('Name', '') or ''),
            cost_type=str(row.get('Type', '') or ''),
            vendor=vendor,
            vendor_normalized=normalize_vendor(vendor),
            amount_cents=int(row.get('amount_cents', 0) or 0),
        ))

    return rows


def df_to_budget_rows(df: pd.DataFrame) -> List[BudgetRow]:
    """Convert DataFrame to list of BudgetRow objects."""
    rows = []

    for _, row in df.iterrows():
        budget_code = str(row.get('Budget Code', '') or '')
        cost_type = str(row.get('Cost Type', '') or '')

        rows.append(BudgetRow(
            budget_code=budget_code,
            base_code=extract_base_code(budget_code),
            description=str(row.get('Budget Code Description', '') or ''),
            cost_type=cost_type,
            cost_type_code=extract_cost_type_code(cost_type),
        ))

    return rows


# =============================================================================
# Batch Processing
# =============================================================================

def compute_all_suggestions(
    direct_costs_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    db: Session,
    unmapped_only: bool = True,
    top_k: int = 3
) -> Dict[int, List[Dict]]:
    """
    Compute suggestions for all direct cost rows.

    Args:
        direct_costs_df: DataFrame of direct costs
        budget_df: DataFrame of budget rows
        db: Database session for loading history
        unmapped_only: If True, only process unmapped rows
        top_k: Number of suggestions per row

    Returns:
        Dict mapping direct_cost_id → list of suggestion dicts
    """
    # Load historical data
    history = load_historical_patterns(db)
    match_counts = load_budget_match_counts(db)

    # Convert DataFrames
    budget_rows = df_to_budget_rows(budget_df)

    # Filter to unmapped if requested
    if unmapped_only and 'mapped_budget_code' in direct_costs_df.columns:
        df = direct_costs_df[direct_costs_df['mapped_budget_code'].isna()]
    else:
        df = direct_costs_df

    dc_rows = df_to_direct_cost_rows(df)

    # Compute suggestions for each row
    results = {}

    for dc in dc_rows:
        suggestions = rank_suggestions(dc, budget_rows, history, match_counts, top_k)
        results[dc.id] = [s.to_dict() for s in suggestions]

    return results


def compute_single_suggestion(
    dc_row: Dict,
    budget_df: pd.DataFrame,
    db: Session,
    top_k: int = 3
) -> List[Dict]:
    """
    Compute suggestions for a single direct cost row.

    Args:
        dc_row: Dict with direct cost data
        budget_df: DataFrame of budget rows
        db: Database session
        top_k: Number of suggestions

    Returns:
        List of suggestion dicts
    """
    # Load historical data
    history = load_historical_patterns(db)
    match_counts = load_budget_match_counts(db)

    # Convert budget DataFrame
    budget_rows = df_to_budget_rows(budget_df)

    # Create DirectCostRow
    cost_code = str(dc_row.get('Cost Code', '') or '')
    vendor = str(dc_row.get('Vendor', '') or '')

    dc = DirectCostRow(
        id=dc_row.get('direct_cost_id', 0),
        cost_code=cost_code,
        base_code=extract_base_code(cost_code),
        name=str(dc_row.get('Name', '') or ''),
        cost_type=str(dc_row.get('Type', '') or ''),
        vendor=vendor,
        vendor_normalized=normalize_vendor(vendor),
        amount_cents=int(dc_row.get('amount_cents', 0) or 0),
    )

    # Compute and return suggestions
    suggestions = rank_suggestions(dc, budget_rows, history, match_counts, top_k)
    return [s.to_dict() for s in suggestions]


# =============================================================================
# Cache Management
# =============================================================================

def cache_suggestions(
    db: Session,
    suggestions: Dict[int, List[Dict]]
) -> int:
    """
    Store computed suggestions in the cache table.

    Args:
        db: Database session
        suggestions: Dict mapping direct_cost_id → list of suggestions

    Returns:
        Number of rows cached
    """
    from sqlalchemy.dialects.sqlite import insert

    cached = 0
    now = datetime.utcnow()

    for dc_id, suggs in suggestions.items():
        top_score = suggs[0]['total_score'] if suggs else 0.0

        # Upsert into cache
        existing = db.query(SuggestionCache).filter(
            SuggestionCache.direct_cost_id == dc_id
        ).first()

        if existing:
            existing.suggestions = json.dumps(suggs)
            existing.top_score = top_score
            existing.computed_at = now
            existing.stale = False
        else:
            cache_entry = SuggestionCache(
                direct_cost_id=dc_id,
                suggestions=json.dumps(suggs),
                top_score=top_score,
                computed_at=now,
                stale=False
            )
            db.add(cache_entry)

        cached += 1

    db.commit()
    return cached


def get_cached_suggestions(
    db: Session,
    dc_ids: List[int]
) -> Dict[int, List[Dict]]:
    """
    Retrieve cached suggestions for given direct cost IDs.
    Returns only non-stale cached entries.
    """
    cached = db.query(SuggestionCache).filter(
        SuggestionCache.direct_cost_id.in_(dc_ids),
        SuggestionCache.stale == False
    ).all()

    results = {}
    for entry in cached:
        try:
            results[entry.direct_cost_id] = json.loads(entry.suggestions)
        except json.JSONDecodeError:
            continue

    return results


def invalidate_cache_for_pattern(
    db: Session,
    vendor_normalized: str,
    base_code: str
) -> int:
    """
    Mark cache entries as stale for rows matching a pattern.
    Called after user saves/overrides a mapping.

    Returns number of rows marked stale.
    """
    # This requires joining with direct costs data, which we don't have
    # as a table. For now, mark all as stale and let background job recompute.
    # In production, this would be more targeted.

    updated = db.query(SuggestionCache).filter(
        SuggestionCache.stale == False
    ).update({'stale': True})

    db.commit()
    return updated


# =============================================================================
# Feedback Recording
# =============================================================================

def record_mapping_feedback(
    db: Session,
    vendor: str,
    name: str,
    selected_budget_code: str,
    suggested_budget_code: Optional[str] = None,
    suggestion_score: Optional[float] = None,
    user_id: Optional[str] = None
) -> MappingFeedback:
    """
    Record a mapping decision for the feedback loop.

    Args:
        db: Database session
        vendor: Original vendor name
        name: Direct cost name
        selected_budget_code: What the user selected
        suggested_budget_code: What the system suggested (if any)
        suggestion_score: Score of the suggestion (if any)
        user_id: User identifier

    Returns:
        Created MappingFeedback record
    """
    vendor_norm = normalize_vendor(vendor)
    name_prefix = extract_name_prefix(name)

    was_override = (
        suggested_budget_code is not None and
        suggested_budget_code != selected_budget_code
    )

    feedback = MappingFeedback(
        vendor_normalized=vendor_norm,
        name_prefix=name_prefix,
        budget_code=selected_budget_code,
        was_override=was_override,
        suggested_budget_code=suggested_budget_code if was_override else None,
        confidence_at_suggestion=suggestion_score,
        user_id=user_id,
        created_at=datetime.utcnow()
    )

    db.add(feedback)

    # Update budget match stats
    update_budget_match_stats(db, selected_budget_code, was_override, suggested_budget_code)

    db.commit()

    return feedback


def update_budget_match_stats(
    db: Session,
    selected_budget_code: str,
    was_override: bool,
    suggested_budget_code: Optional[str]
):
    """Update match statistics for budget codes involved in a mapping."""
    now = datetime.utcnow()

    # Increment selected budget's match count
    selected_stats = db.query(BudgetMatchStats).filter(
        BudgetMatchStats.budget_code == selected_budget_code
    ).first()

    if selected_stats:
        selected_stats.total_matches += 1
        selected_stats.last_updated = now
    else:
        selected_stats = BudgetMatchStats(
            budget_code=selected_budget_code,
            total_matches=1,
            override_count=0,
            trust_score=1.0,
            last_updated=now
        )
        db.add(selected_stats)

    # If override, decrement suggested budget's trust score
    if was_override and suggested_budget_code:
        suggested_stats = db.query(BudgetMatchStats).filter(
            BudgetMatchStats.budget_code == suggested_budget_code
        ).first()

        if suggested_stats:
            suggested_stats.override_count += 1
            # Decay trust score (minimum 0.5)
            suggested_stats.trust_score = max(
                0.5,
                suggested_stats.trust_score * 0.95
            )
            suggested_stats.last_updated = now

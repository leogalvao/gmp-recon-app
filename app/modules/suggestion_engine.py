"""
Suggestion Engine for Direct Cost → Budget Mapping.

Implements the match scoring algorithm with configurable weights from YAML:
- Code prefix matching (default 40% weight)
- Cost type matching (default 20% weight)
- Fuzzy text similarity (default 30% weight)
- Historical pattern boost (default 10% weight)

Confidence bands (configurable):
- High: ≥ 0.85 (auto-suggest, green highlight)
- Medium: 0.60 – 0.84 (show suggestion, yellow highlight)
- Low: < 0.60 (no suggestion, manual required)

Configuration is loaded from gmp_mapping_config.yaml via app.config.
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
from ..config import get_config


# =============================================================================
# Configuration Accessors
# =============================================================================

def _get_weights() -> Dict[str, float]:
    """Get scoring weights from config."""
    return get_config().suggestion_weights


def _get_thresholds() -> Dict[str, float]:
    """Get confidence thresholds from config."""
    config = get_config()
    bands = config.confidence_bands
    return {
        'high': bands.get('high', {}).get('min_score', 0.85),
        'medium': bands.get('medium', {}).get('min_score', 0.60),
        'low': bands.get('low', {}).get('min_score', 0.40),
    }


def _get_type_score(source_type: str, target_type: str) -> float:
    """Get cost type compatibility score from config."""
    return get_config().get_cost_type_score(source_type, target_type)


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
# Blocking Index for O(n×b) Fuzzy Matching
# =============================================================================

class BudgetBlockingIndex:
    """
    Blocking index for efficient budget row lookup.

    Instead of comparing each direct cost to ALL budget rows O(n×m),
    we first look up budget rows in the same "block" (by base_code).
    This reduces complexity to O(n×b) where b = average block size.

    Blocking strategy:
    1. Primary block: base_code (e.g., "4-010" → [budget rows with same prefix])
    2. Secondary block: cost_type_code (e.g., "L" → [labor budget rows])
    3. Fallback: sample of all rows (for items with no code match)
    """

    def __init__(self, budget_rows: List[BudgetRow], fallback_sample_size: int = 50):
        self.all_rows = budget_rows
        self.fallback_sample_size = fallback_sample_size

        # Primary index: base_code → List[BudgetRow]
        self.by_base_code: Dict[str, List[BudgetRow]] = {}

        # Secondary index: cost_type_code → List[BudgetRow]
        self.by_type: Dict[str, List[BudgetRow]] = {}

        # Build indices
        for row in budget_rows:
            # Primary index
            if row.base_code:
                if row.base_code not in self.by_base_code:
                    self.by_base_code[row.base_code] = []
                self.by_base_code[row.base_code].append(row)

            # Secondary index
            if row.cost_type_code:
                if row.cost_type_code not in self.by_type:
                    self.by_type[row.cost_type_code] = []
                self.by_type[row.cost_type_code].append(row)

    def get_candidates(self, dc_base_code: str, dc_type_code: str) -> List[BudgetRow]:
        """
        Get candidate budget rows for a direct cost using blocking.

        Priority:
        1. Same base_code (highest priority, code_match_score = 1.0)
        2. Same cost_type_code (medium priority, likely same trade)
        3. Fallback sample (ensure we don't miss good matches)

        Returns deduplicated list of candidates.
        """
        candidates = []
        seen_codes = set()

        # 1. Primary block: same base_code
        if dc_base_code and dc_base_code in self.by_base_code:
            for row in self.by_base_code[dc_base_code]:
                if row.budget_code not in seen_codes:
                    candidates.append(row)
                    seen_codes.add(row.budget_code)

        # 2. Secondary block: same cost type
        if dc_type_code and dc_type_code in self.by_type:
            for row in self.by_type[dc_type_code]:
                if row.budget_code not in seen_codes:
                    candidates.append(row)
                    seen_codes.add(row.budget_code)

        # 3. Fallback: sample of remaining rows if we have few candidates
        if len(candidates) < self.fallback_sample_size:
            remaining = self.fallback_sample_size - len(candidates)
            for row in self.all_rows[:remaining * 2]:  # Oversample to find unique
                if row.budget_code not in seen_codes:
                    candidates.append(row)
                    seen_codes.add(row.budget_code)
                    if len(candidates) >= self.fallback_sample_size:
                        break

        return candidates


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
    Compute cost type match score using compatibility matrix from config.
    Returns 1.0 for exact match, partial score for compatible types, default otherwise.
    """
    dc_code = extract_cost_type_code(dc_type)
    budget_code = extract_cost_type_code(budget_type)

    if not dc_code or not budget_code:
        return 0.0

    # Use config-based cost type scoring
    return _get_type_score(dc_code, budget_code)


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


# =============================================================================
# Bayesian Scoring with Trust Decay
# =============================================================================

# Loss aversion coefficient (Kahneman & Tversky: losses hurt ~2x gains)
LOSS_AVERSION_LAMBDA = 1.5

# Trust decay parameters
TRUST_DECAY_RATE = 0.1  # How much trust decreases per override
MIN_TRUST = 0.3  # Floor for trust score


def compute_bayesian_trust(
    total_matches: int,
    override_count: int,
    prior_trust: float = 1.0
) -> float:
    """
    Compute Bayesian trust score with loss aversion.

    Model:
    - Prior: Initial trust = 1.0 (uninformative)
    - Likelihood: Based on observed accept/override ratio
    - Loss aversion: Overrides penalized λ times more than accepts reward

    Formula:
        trust = prior × (accepts + 1) / (accepts + λ × overrides + 2)

    Where:
        - accepts = total_matches - override_count
        - +1, +2 are Laplace smoothing terms
    """
    accepts = max(0, total_matches - override_count)

    # Bayesian update with loss aversion
    # Numerator: accepts + prior (Laplace smoothing)
    # Denominator: total effective observations
    effective_overrides = LOSS_AVERSION_LAMBDA * override_count

    trust = (accepts + 1) / (accepts + effective_overrides + 2)

    # Apply prior and floor
    trust = max(MIN_TRUST, prior_trust * trust)

    return trust


def compute_trust_adjusted_score(
    base_score: float,
    budget_code: str,
    trust_scores: Dict[str, float]
) -> float:
    """
    Adjust base score using Bayesian trust score.

    High trust (few overrides): score unchanged
    Low trust (many overrides): score penalized

    Formula: adjusted_score = base_score × trust_score
    """
    trust = trust_scores.get(budget_code, 1.0)
    return base_score * trust


def update_trust_on_feedback(
    db: Session,
    budget_code: str,
    was_accepted: bool,
    was_override: bool
) -> float:
    """
    Update trust score after user feedback.

    Args:
        db: Database session
        budget_code: Budget code that was suggested
        was_accepted: True if user accepted this suggestion
        was_override: True if user chose different from top suggestion

    Returns:
        Updated trust score
    """
    stats = db.query(BudgetMatchStats).filter(
        BudgetMatchStats.budget_code == budget_code
    ).first()

    if not stats:
        stats = BudgetMatchStats(
            budget_code=budget_code,
            total_matches=0,
            override_count=0,
            trust_score=1.0
        )
        db.add(stats)

    # Update counts
    if was_accepted:
        stats.total_matches += 1
    if was_override:
        stats.override_count += 1

    # Recompute trust score
    stats.trust_score = compute_bayesian_trust(
        stats.total_matches,
        stats.override_count,
        prior_trust=1.0
    )

    db.commit()
    return stats.trust_score


def compute_match_score(
    dc: DirectCostRow,
    budget: BudgetRow,
    history: Dict[Tuple[str, str], str],
    trust_scores: Optional[Dict[str, float]] = None
) -> ScoredMatch:
    """
    Compute composite match score for a direct cost → budget pair.

    Score formula (weights from config):
        base_score = (code_prefix_match × code_match) + (cost_type_match × type_match) +
                     (text_similarity × text_sim) + (historical_pattern × historical_boost)

    With Bayesian trust adjustment:
        final_score = base_score × trust_score

    Where trust_score decreases with user overrides (loss aversion λ=1.5).
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

    # Get weights from config
    weights = _get_weights()

    # Weighted sum using config weights (base score)
    base_score = (
        weights.get('code_prefix_match', 0.40) * code_match +
        weights.get('cost_type_match', 0.20) * type_match +
        weights.get('text_similarity', 0.30) * text_sim +
        weights.get('historical_pattern', 0.10) * historical
    )

    # Apply Bayesian trust adjustment if trust scores provided
    if trust_scores is not None:
        total_score = compute_trust_adjusted_score(base_score, budget.budget_code, trust_scores)
    else:
        total_score = base_score

    # Determine confidence band using config thresholds
    thresholds = _get_thresholds()
    if total_score >= thresholds['high']:
        confidence_band = 'high'
    elif total_score >= thresholds['medium']:
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
    top_k: int = 3,
    trust_scores: Optional[Dict[str, float]] = None
) -> List[ScoredMatch]:
    """
    Rank all budget rows for a direct cost and return top-k suggestions.

    Tie-breaking rules (when scores are equal):
    1. Prefer budget code with more historical matches
    2. Alphabetical by budget code (deterministic fallback)

    If trust_scores provided, applies Bayesian trust adjustment (penalizes
    frequently-overridden suggestions).
    """
    candidates = []
    thresholds = _get_thresholds()
    min_threshold = thresholds.get('low', 0.40)

    for budget in budget_rows:
        scored = compute_match_score(dc, budget, history, trust_scores)

        # Only consider candidates above minimum threshold from config
        if scored.total_score >= min_threshold:
            scored.historical_match_count = match_counts.get(budget.budget_code, 0)
            candidates.append(scored)

    # Sort by: score DESC, historical_count DESC, budget_code ASC
    candidates.sort(key=lambda x: (
        -x.total_score,
        -x.historical_match_count,
        x.budget_code
    ))

    return candidates[:top_k]


def rank_suggestions_blocked(
    dc: DirectCostRow,
    blocking_index: BudgetBlockingIndex,
    history: Dict[Tuple[str, str], str],
    match_counts: Dict[str, int],
    top_k: int = 3,
    trust_scores: Optional[Dict[str, float]] = None
) -> List[ScoredMatch]:
    """
    Rank budget rows for a direct cost using blocking for efficiency.

    Uses BudgetBlockingIndex to reduce candidate set from O(m) to O(b),
    where b is the average block size (much smaller than total budget rows).

    If trust_scores provided, applies Bayesian trust adjustment.
    """
    # Get candidate budget rows from blocking index
    dc_type_code = extract_cost_type_code(dc.cost_type)
    candidates_budget = blocking_index.get_candidates(dc.base_code, dc_type_code)

    candidates = []
    thresholds = _get_thresholds()
    min_threshold = thresholds.get('low', 0.40)

    for budget in candidates_budget:
        scored = compute_match_score(dc, budget, history, trust_scores)

        if scored.total_score >= min_threshold:
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
    top_k: int = 3,
    use_blocking: bool = True,
    use_bayesian_trust: bool = True
) -> Dict[int, List[Dict]]:
    """
    Compute suggestions for all direct cost rows.

    Args:
        direct_costs_df: DataFrame of direct costs
        budget_df: DataFrame of budget rows
        db: Database session for loading history
        unmapped_only: If True, only process unmapped rows
        top_k: Number of suggestions per row
        use_blocking: If True, use blocking index for O(n×b) performance
        use_bayesian_trust: If True, apply trust score adjustment (penalizes overridden suggestions)

    Returns:
        Dict mapping direct_cost_id → list of suggestion dicts
    """
    # Load historical data
    history = load_historical_patterns(db)
    match_counts = load_budget_match_counts(db)

    # Load trust scores for Bayesian adjustment
    trust_scores = load_budget_trust_scores(db) if use_bayesian_trust else None

    # Convert DataFrames
    budget_rows = df_to_budget_rows(budget_df)

    # Build blocking index for O(n×b) performance
    blocking_index = BudgetBlockingIndex(budget_rows) if use_blocking else None

    # Filter to unmapped if requested
    if unmapped_only and 'mapped_budget_code' in direct_costs_df.columns:
        df = direct_costs_df[direct_costs_df['mapped_budget_code'].isna()]
    else:
        df = direct_costs_df

    dc_rows = df_to_direct_cost_rows(df)

    # Compute suggestions for each row
    results = {}

    for dc in dc_rows:
        if use_blocking and blocking_index:
            suggestions = rank_suggestions_blocked(
                dc, blocking_index, history, match_counts, top_k, trust_scores
            )
        else:
            suggestions = rank_suggestions(
                dc, budget_rows, history, match_counts, top_k, trust_scores
            )
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

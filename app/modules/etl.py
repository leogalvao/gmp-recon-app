"""
ETL Module for GMP Reconciliation App.
Handles loading, parsing, and normalizing data from GMP, Budget, and Direct Cost files.
All monetary values converted to integer cents using Decimal for precision.
"""
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union, List, Tuple
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


DATA_DIR = Path("./data")


def parse_money_to_cents(value: Union[str, float, int, None]) -> int:
    """
    Parse currency strings to integer cents.
    CRITICAL: Uses Decimal internally to avoid float precision loss.

    Handles:
        " 715,643.50 " → 71564350
        "$1,234.56"    → 123456
        "-$500.00"     → -50000
        "($1,000.00)"  → -100000 (accounting negative)
        " -   " or "-" → 0
        None, NaN, ""  → 0
        715643.50      → 71564350 (float/int input)

    Returns:
        int: Amount in cents, suitable for penny-perfect arithmetic
    """
    # Handle null/empty
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0

    if isinstance(value, str):
        s = value.strip()
        if s == '' or s == '-':
            return 0

    # Handle numeric types directly with Decimal
    if isinstance(value, (int, float)):
        try:
            d = Decimal(str(value))
            cents = (d * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            return int(cents)
        except (InvalidOperation, ValueError):
            return 0

    # String parsing
    s = str(value).strip()

    # Detect negative (prefix '-', suffix '-', or parentheses for accounting notation)
    negative = False
    if s.startswith('-') or s.endswith('-') or (s.startswith('(') and s.endswith(')')):
        negative = True

    # Remove all non-numeric characters except decimal point
    s = re.sub(r'[^\d.]', '', s)

    if s == '' or s == '.':
        return 0

    # Handle multiple decimal points (malformed input)
    if s.count('.') > 1:
        return 0

    try:
        d = Decimal(s)
        cents = (d * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        return -int(cents) if negative else int(cents)
    except (InvalidOperation, ValueError):
        return 0


def cents_to_display(cents: int) -> str:
    """Format integer cents as USD display string."""
    if cents < 0:
        return f"-${abs(cents)/100:,.2f}"
    return f"${cents/100:,.2f}"


def allocate_largest_remainder(total_cents: int, weights: list[float]) -> list[int]:
    """
    Allocate total_cents to buckets using Largest Remainder Method (Hamilton's Method).
    Guarantees sum(result) == total_cents exactly - no penny drift.

    Args:
        total_cents: Total amount to allocate (integer cents)
        weights: List of weights (percentages as 0.0-1.0, must sum to 1.0)

    Returns:
        List of integer cents, one per weight, summing exactly to total_cents

    Example:
        allocate_largest_remainder(10000, [0.333, 0.667])
        → [3330, 6670]  # Sums to 10000 exactly
    """
    if not weights:
        return []

    if len(weights) == 1:
        return [total_cents]

    # Normalize weights to ensure they sum to 1.0
    weight_sum = sum(weights)
    if weight_sum == 0:
        # Equal split when all weights are zero
        base = total_cents // len(weights)
        result = [base] * len(weights)
        result[0] += total_cents - sum(result)
        return result

    normalized = [w / weight_sum for w in weights]

    # Step 1: Calculate exact allocations and floor values
    exact = [total_cents * w for w in normalized]
    floored = [int(e) for e in exact]

    # Step 2: Calculate remainders
    remainders = [e - f for e, f in zip(exact, floored)]

    # Step 3: Distribute leftover cents to items with largest remainders
    leftover = total_cents - sum(floored)

    # Get indices sorted by remainder (descending)
    sorted_indices = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)

    # Distribute one cent to each of the top 'leftover' positions
    for i in range(int(leftover)):
        floored[sorted_indices[i]] += 1

    return floored


def allocate_east_west(total_cents: int, pct_east: float, pct_west: float) -> tuple[int, int]:
    """
    Allocate total_cents between East and West using Largest Remainder Method.

    Args:
        total_cents: Total amount to split
        pct_east: East percentage (0.0-1.0)
        pct_west: West percentage (0.0-1.0)

    Returns:
        Tuple of (east_cents, west_cents) that sum exactly to total_cents

    Example:
        allocate_east_west(10000, 0.333, 0.667)
        → (3330, 6670)
    """
    result = allocate_largest_remainder(total_cents, [pct_east, pct_west])
    return (result[0], result[1]) if len(result) == 2 else (0, 0)


def normalize_code(code: str) -> str:
    """
    Normalize cost/budget codes to consistent format.
    E.g., '1-010.DB' -> '1-010', extract just the base code.
    """
    if pd.isna(code) or code is None:
        return ''
    code = str(code).strip()
    # Extract base code (before any dot suffix)
    match = re.match(r'^(\d+-\d+)', code)
    if match:
        return match.group(1)
    return code


def normalize_invoice_number(inv: str) -> str:
    """
    Normalize invoice numbers for duplicate detection.
    Strip spaces, hyphens, casefold.
    """
    if pd.isna(inv) or inv is None:
        return ''
    return str(inv).replace(' ', '').replace('-', '').lower().strip()


def extract_division_key(cost_code_tier2: str) -> str:
    """
    Extract division key from Cost Code Tier 2.
    E.g., '1-010 - General Conditions' -> '1'
    """
    if pd.isna(cost_code_tier2):
        return ''
    match = re.match(r'^(\d+)', str(cost_code_tier2).strip())
    if match:
        return match.group(1)
    return ''


def file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    if not path.exists():
        return ''
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_gmp_xlsx(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load GMP Excel file.
    Returns DataFrame with columns: [GMP, amount_total_cents]
    """
    path = path or DATA_DIR / "GMP-Amount.xlsx"
    df = pd.read_excel(path)
    
    # Ensure expected columns
    if 'GMP' not in df.columns or 'Amount Total' not in df.columns:
        raise ValueError(f"GMP file must have 'GMP' and 'Amount Total' columns. Found: {df.columns.tolist()}")
    
    df = df[['GMP', 'Amount Total']].copy()
    df['amount_total_cents'] = df['Amount Total'].apply(parse_money_to_cents)
    df['gmp_id'] = df.index
    
    return df


def load_budget_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Procore budget export CSV.
    Returns DataFrame with normalized codes and amounts in cents.
    """
    path = path or DATA_DIR / "budget.csv"
    df = pd.read_csv(path, encoding='utf-8-sig')
    
    # Expected key columns
    expected = ['Cost Code Tier 1', 'Cost Code Tier 2', 'Budget Code', 
                'Budget Code Description', 'Current Budget', 'Committed Costs', 'Direct Cost Tool']
    
    for col in expected:
        if col not in df.columns:
            df[col] = None
    
    # Convert money columns to cents
    for col in ['Current Budget', 'Committed Costs', 'Direct Cost Tool']:
        df[f'{col.lower().replace(" ", "_")}_cents'] = df[col].apply(parse_money_to_cents)
    
    # Add normalized base code
    df['base_code'] = df['Budget Code'].apply(normalize_code)
    df['division_key'] = df['Cost Code Tier 2'].apply(extract_division_key)
    df['budget_id'] = df.index
    
    return df


def load_direct_costs_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Procore direct costs export CSV.
    Returns DataFrame with normalized codes, amounts in cents, and parsed dates.
    """
    path = path or DATA_DIR / "direct_costs_by_group.csv"
    df = pd.read_csv(path, encoding='utf-8-sig')
    
    # Handle duplicate 'Type' column (known issue in the data)
    if df.columns.tolist().count('Type') > 1:
        cols = df.columns.tolist()
        new_cols = []
        type_count = 0
        for c in cols:
            if c == 'Type':
                type_count += 1
                new_cols.append(f'Type_{type_count}' if type_count > 1 else 'Type')
            else:
                new_cols.append(c)
        df.columns = new_cols
    
    # Expected columns
    expected = ['Cost Code', 'Name', 'Type', 'Date', 'Vendor', 'Invoice #', 'Amount', 'Description']
    for col in expected:
        if col not in df.columns:
            df[col] = None
    
    # Parse Amount to cents
    df['amount_cents'] = df['Amount'].apply(parse_money_to_cents)
    
    # Parse Date
    df['date_parsed'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    
    # Normalize codes
    df['base_code'] = df['Cost Code'].apply(normalize_code)
    df['normalized_invoice'] = df['Invoice #'].apply(normalize_invoice_number)
    
    # Clean vendor names
    df['vendor_clean'] = df['Vendor'].fillna('').str.strip().str.lower()
    
    # Add row ID for tracking
    df['direct_cost_id'] = df.index
    
    # Default exclusion flag
    df['excluded_from_actuals'] = False
    
    return df


def load_allocations_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load allocations CSV (West/East splits).
    Returns DataFrame with allocation rules, or empty DataFrame if file doesn't exist.
    """
    path = path or DATA_DIR / "allocations.csv"

    if not path.exists():
        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=['code', 'region', 'pct_west', 'pct_east', 'confirmed'])

    df = pd.read_csv(path)

    # Validate schema
    required = ['code', 'region', 'pct_west', 'pct_east']
    for col in required:
        if col not in df.columns:
            df[col] = 0.5 if 'pct' in col else 'Both' if col == 'region' else ''

    if 'confirmed' not in df.columns:
        df['confirmed'] = True

    # Validate: pct_west + pct_east must equal 1.0
    df['_sum'] = df['pct_west'] + df['pct_east']
    invalid = df[abs(df['_sum'] - 1.0) > 0.001]
    if len(invalid) > 0:
        print(f"Warning: {len(invalid)} allocation rows have pct_west + pct_east != 1.0")
    df = df.drop(columns=['_sum'])

    return df


def load_breakdown_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load owner's East/West breakdown CSV.
    Returns DataFrame with columns suitable for GMPBudgetBreakdown model.

    Expected CSV columns (flexible naming):
        - cost_code_description / Description / Item
        - gmp_sov / GMP SOV / Total / Amount
        - east_funded / East / East Funded
        - west_funded / West / West Funded

    Returns DataFrame with:
        - cost_code_description: str
        - gmp_sov_cents: int
        - east_funded_cents: int
        - west_funded_cents: int
        - pct_east: float (0.0-1.0)
        - pct_west: float (0.0-1.0)
    """
    path = path or DATA_DIR / "breakdown.csv"

    if not path.exists():
        return pd.DataFrame(columns=[
            'cost_code_description', 'gmp_sov_cents', 'east_funded_cents',
            'west_funded_cents', 'pct_east', 'pct_west'
        ])

    df = pd.read_csv(path, encoding='utf-8-sig')

    # Column name mapping (flexible)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('cost_code_description', 'description', 'item', 'name', 'cost code description', 'cost code - description'):
            col_map['cost_code_description'] = col
        elif col_lower in ('gmp_sov', 'gmp sov', 'total', 'amount', 'sov'):
            col_map['gmp_sov'] = col
        elif col_lower in ('east_funded', 'east funded', 'east', 'east addition funded on gmp'):
            col_map['east_funded'] = col
        elif col_lower in ('west_funded', 'west funded', 'west', 'west addition funded on gmp'):
            col_map['west_funded'] = col

    # Validate required columns
    required = ['cost_code_description', 'gmp_sov']
    for req in required:
        if req not in col_map:
            raise ValueError(f"Breakdown CSV missing required column: {req}. Found: {df.columns.tolist()}")

    # Build result DataFrame
    result = pd.DataFrame()
    result['cost_code_description'] = df[col_map['cost_code_description']].fillna('').astype(str).str.strip()
    result['gmp_sov_cents'] = df[col_map['gmp_sov']].apply(parse_money_to_cents)

    # East/West columns are optional - if missing, assume 50/50 split
    if 'east_funded' in col_map:
        result['east_funded_cents'] = df[col_map['east_funded']].apply(parse_money_to_cents)
    else:
        result['east_funded_cents'] = result['gmp_sov_cents'] // 2

    if 'west_funded' in col_map:
        result['west_funded_cents'] = df[col_map['west_funded']].apply(parse_money_to_cents)
    else:
        result['west_funded_cents'] = result['gmp_sov_cents'] - result['east_funded_cents']

    # Calculate percentages (avoid division by zero)
    result['pct_east'] = result.apply(
        lambda r: r['east_funded_cents'] / r['gmp_sov_cents'] if r['gmp_sov_cents'] > 0 else 0.5,
        axis=1
    )
    result['pct_west'] = result.apply(
        lambda r: r['west_funded_cents'] / r['gmp_sov_cents'] if r['gmp_sov_cents'] > 0 else 0.5,
        axis=1
    )

    # Filter out rows with zero or negative SOV
    result = result[result['gmp_sov_cents'] > 0].copy()

    # Add row ID
    result['breakdown_id'] = result.index

    return result


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy matching.
    Strips common words, lowercases, removes special characters.
    """
    if not text:
        return ''

    # Lowercase
    text = text.lower().strip()

    # Remove common prefixes/suffixes that don't help matching
    noise_words = [
        'division', 'div', 'work', 'general', 'project', 'total',
        'subtotal', 'sub-total', 'summary', 'estimate'
    ]
    for word in noise_words:
        text = re.sub(rf'\b{word}\b', '', text)

    # Remove special characters except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def fuzzy_match_breakdown_to_gmp(
    breakdown_df: pd.DataFrame,
    gmp_df: pd.DataFrame,
    score_cutoff: int = 60
) -> pd.DataFrame:
    """
    Fuzzy match breakdown cost code descriptions to GMP division names.

    Args:
        breakdown_df: DataFrame from load_breakdown_csv()
        gmp_df: DataFrame from load_gmp_xlsx()
        score_cutoff: Minimum match score (0-100) to accept match

    Returns:
        DataFrame with breakdown data plus matched GMP division and score
    """
    if not RAPIDFUZZ_AVAILABLE:
        # Fallback: no fuzzy matching, return breakdown with empty matches
        breakdown_df = breakdown_df.copy()
        breakdown_df['gmp_division'] = None
        breakdown_df['match_score'] = 0
        return breakdown_df

    if breakdown_df.empty or gmp_df.empty:
        breakdown_df = breakdown_df.copy()
        breakdown_df['gmp_division'] = None
        breakdown_df['match_score'] = 0
        return breakdown_df

    # Get GMP division names
    gmp_divisions = gmp_df['GMP'].dropna().unique().tolist()

    # Prepare normalized versions for matching
    gmp_normalized = {normalize_for_matching(div): div for div in gmp_divisions}
    gmp_keys = list(gmp_normalized.keys())

    result = breakdown_df.copy()
    matches = []
    scores = []

    for _, row in breakdown_df.iterrows():
        desc = row.get('cost_code_description', '')
        desc_normalized = normalize_for_matching(desc)

        if not desc_normalized or not gmp_keys:
            matches.append(None)
            scores.append(0)
            continue

        # Use RapidFuzz to find best match
        best = process.extractOne(
            desc_normalized,
            gmp_keys,
            scorer=fuzz.token_set_ratio,
            score_cutoff=score_cutoff
        )

        if best:
            matched_key, score, _ = best
            original_gmp = gmp_normalized[matched_key]
            matches.append(original_gmp)
            scores.append(int(score))
        else:
            matches.append(None)
            scores.append(0)

    result['gmp_division'] = matches
    result['match_score'] = scores

    return result


def match_single_to_gmp(
    description: str,
    gmp_divisions: List[str],
    score_cutoff: int = 60
) -> Tuple[Optional[str], int]:
    """
    Match a single description to GMP divisions.

    Args:
        description: Text to match
        gmp_divisions: List of GMP division names
        score_cutoff: Minimum match score (0-100)

    Returns:
        Tuple of (matched_division, score) or (None, 0) if no match
    """
    if not RAPIDFUZZ_AVAILABLE or not description or not gmp_divisions:
        return (None, 0)

    desc_normalized = normalize_for_matching(description)
    gmp_normalized = {normalize_for_matching(div): div for div in gmp_divisions}
    gmp_keys = list(gmp_normalized.keys())

    best = process.extractOne(
        desc_normalized,
        gmp_keys,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff
    )

    if best:
        matched_key, score, _ = best
        return (gmp_normalized[matched_key], int(score))

    return (None, 0)


def get_max_transaction_date(direct_costs_df: pd.DataFrame) -> datetime:
    """Get the maximum transaction date from direct costs for as_of_date default."""
    valid_dates = direct_costs_df['date_parsed'].dropna()
    if len(valid_dates) == 0:
        return datetime.now()
    return valid_dates.max().to_pydatetime()


# =============================================================================
# Schedule ETL Module (Task 6)
# =============================================================================

def load_schedule_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load project schedule CSV export (from P6, MS Project, or similar).
    Returns DataFrame suitable for ScheduleActivity model.

    Expected CSV columns (flexible naming):
        - Task Name / Activity Name / Name
        - Activity ID / Task ID / ID
        - WBS / WBS Code
        - % Complete / Percent Complete / Progress
        - UID / Unique ID / GUID (optional)

    Returns DataFrame with:
        - row_number: int (original row order)
        - task_name: str
        - source_uid: str (unique identifier if available)
        - activity_id: str
        - wbs: str
        - pct_complete: int (0-100)
    """
    path = path or DATA_DIR / "schedule.csv"

    if not path.exists():
        return pd.DataFrame(columns=[
            'row_number', 'task_name', 'source_uid', 'activity_id', 'wbs', 'pct_complete'
        ])

    df = pd.read_csv(path, encoding='utf-8-sig')

    # Column name mapping (flexible)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('task name', 'activity name', 'name', 'task_name', 'activity_name'):
            col_map['task_name'] = col
        elif col_lower in ('activity id', 'task id', 'id', 'activity_id', 'task_id'):
            col_map['activity_id'] = col
        elif col_lower in ('wbs', 'wbs code', 'wbs_code'):
            col_map['wbs'] = col
        elif col_lower in ('% complete', 'percent complete', 'progress', 'pct_complete', 'complete'):
            col_map['pct_complete'] = col
        elif col_lower in ('uid', 'unique id', 'guid', 'source_uid', 'unique_id'):
            col_map['source_uid'] = col

    # Validate required columns
    if 'task_name' not in col_map:
        raise ValueError(f"Schedule CSV missing task name column. Found: {df.columns.tolist()}")

    # Build result DataFrame
    result = pd.DataFrame()
    result['row_number'] = range(len(df))
    result['task_name'] = df[col_map['task_name']].fillna('').astype(str).str.strip()

    # Optional columns
    if 'source_uid' in col_map:
        result['source_uid'] = df[col_map['source_uid']].fillna('').astype(str).str.strip()
        # Replace empty strings with None for uniqueness
        result['source_uid'] = result['source_uid'].replace('', None)
    else:
        result['source_uid'] = None

    if 'activity_id' in col_map:
        result['activity_id'] = df[col_map['activity_id']].fillna('').astype(str).str.strip()
    else:
        result['activity_id'] = ''

    if 'wbs' in col_map:
        result['wbs'] = df[col_map['wbs']].fillna('').astype(str).str.strip()
    else:
        result['wbs'] = ''

    if 'pct_complete' in col_map:
        # Parse percentage - handle both "50%" and "50" formats
        def parse_pct(val):
            if pd.isna(val):
                return 0
            s = str(val).strip().replace('%', '')
            try:
                pct = float(s)
                # If value > 1, assume it's already in 0-100 scale
                # If value <= 1, assume it's in 0-1 scale
                if 0 <= pct <= 1:
                    return int(pct * 100)
                return int(min(100, max(0, pct)))
            except ValueError:
                return 0

        result['pct_complete'] = df[col_map['pct_complete']].apply(parse_pct)
    else:
        result['pct_complete'] = 0

    # Filter out empty task names
    result = result[result['task_name'].str.len() > 0].copy()

    return result


def match_schedule_to_gmp(
    schedule_df: pd.DataFrame,
    gmp_df: pd.DataFrame,
    score_cutoff: int = 50
) -> pd.DataFrame:
    """
    Fuzzy match schedule task names to GMP divisions.

    Args:
        schedule_df: DataFrame from load_schedule_csv()
        gmp_df: DataFrame from load_gmp_xlsx()
        score_cutoff: Minimum match score (0-100)

    Returns:
        DataFrame with schedule data plus matched GMP division and score
    """
    if not RAPIDFUZZ_AVAILABLE:
        schedule_df = schedule_df.copy()
        schedule_df['gmp_division'] = None
        schedule_df['match_score'] = 0
        return schedule_df

    if schedule_df.empty or gmp_df.empty:
        schedule_df = schedule_df.copy()
        schedule_df['gmp_division'] = None
        schedule_df['match_score'] = 0
        return schedule_df

    gmp_divisions = gmp_df['GMP'].dropna().unique().tolist()

    result = schedule_df.copy()
    matches = []
    scores = []

    for _, row in schedule_df.iterrows():
        task = row.get('task_name', '')
        # Combine task name and WBS for better matching
        wbs = row.get('wbs', '')
        combined = f"{task} {wbs}".strip()

        matched_div, score = match_single_to_gmp(combined, gmp_divisions, score_cutoff)
        matches.append(matched_div)
        scores.append(score)

    result['gmp_division'] = matches
    result['match_score'] = scores

    return result


def calculate_weighted_progress(
    schedule_df: pd.DataFrame,
    gmp_division: str
) -> float:
    """
    Calculate weighted progress for a GMP division from schedule activities.

    Args:
        schedule_df: DataFrame with matched schedule activities
        gmp_division: GMP division to calculate progress for

    Returns:
        Weighted average progress (0.0 - 1.0) for the division
    """
    # Filter to activities mapped to this division
    matched = schedule_df[schedule_df['gmp_division'] == gmp_division]

    if matched.empty:
        return 0.0

    # Use match score as weight (higher confidence matches count more)
    weights = matched['match_score'].values
    progress = matched['pct_complete'].values / 100.0

    weight_sum = weights.sum()
    if weight_sum == 0:
        # Equal weighting if all scores are 0
        return progress.mean()

    return (progress * weights).sum() / weight_sum


def get_file_hashes() -> Dict[str, str]:
    """Get hashes of all input files for change detection."""
    return {
        'gmp': file_hash(DATA_DIR / "GMP-Amount.xlsx"),
        'budget': file_hash(DATA_DIR / "budget.csv"),
        'direct_costs': file_hash(DATA_DIR / "direct_costs_by_group.csv"),
        'allocations': file_hash(DATA_DIR / "allocations.csv") if (DATA_DIR / "allocations.csv").exists() else '',
        'breakdown': file_hash(DATA_DIR / "breakdown.csv") if (DATA_DIR / "breakdown.csv").exists() else '',
        'schedule': file_hash(DATA_DIR / "schedule.csv") if (DATA_DIR / "schedule.csv").exists() else ''
    }


class DataLoader:
    """
    Singleton-like data loader that caches DataFrames and tracks changes.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cache = {}
        self._file_hashes = {}
        self.reload()
    
    def reload(self):
        """Reload all data files."""
        self._file_hashes = get_file_hashes()
        self._cache['gmp'] = load_gmp_xlsx()
        self._cache['budget'] = load_budget_csv()
        self._cache['direct_costs'] = load_direct_costs_csv()
        self._cache['allocations'] = load_allocations_csv()
        self._cache['breakdown'] = load_breakdown_csv()
        self._cache['schedule'] = load_schedule_csv()
        self._cache['max_date'] = get_max_transaction_date(self._cache['direct_costs'])
    
    def check_for_changes(self) -> bool:
        """Check if any input files have changed."""
        current_hashes = get_file_hashes()
        return current_hashes != self._file_hashes
    
    @property
    def gmp(self) -> pd.DataFrame:
        return self._cache.get('gmp', pd.DataFrame())
    
    @property
    def budget(self) -> pd.DataFrame:
        return self._cache.get('budget', pd.DataFrame())
    
    @property
    def direct_costs(self) -> pd.DataFrame:
        return self._cache.get('direct_costs', pd.DataFrame())
    
    @property
    def allocations(self) -> pd.DataFrame:
        return self._cache.get('allocations', pd.DataFrame())

    @property
    def breakdown(self) -> pd.DataFrame:
        return self._cache.get('breakdown', pd.DataFrame())

    @property
    def schedule(self) -> pd.DataFrame:
        return self._cache.get('schedule', pd.DataFrame())

    @property
    def max_transaction_date(self) -> datetime:
        return self._cache.get('max_date', datetime.now())


# Module-level convenience function
def get_data_loader() -> DataLoader:
    return DataLoader()

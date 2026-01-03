"""
ETL Module for GMP Reconciliation App.
Handles loading, parsing, and normalizing data from GMP, Budget, and Direct Cost files.
All monetary values converted to integer cents.
"""
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import hashlib


DATA_DIR = Path("./data")


def parse_money_to_cents(value) -> int:
    """
    Convert money string or float to integer cents.
    Handles formats like '$1,234.56', '1234.56', '-$500', etc.
    """
    if pd.isna(value) or value == '':
        return 0
    
    if isinstance(value, (int, float)):
        return int(round(float(value) * 100))
    
    # String parsing
    s = str(value).strip()
    
    # Detect negative (could be prefix '-' or parentheses)
    negative = False
    if s.startswith('-') or s.startswith('('):
        negative = True
        s = s.replace('-', '').replace('(', '').replace(')', '')
    
    # Remove currency symbols and commas
    s = s.replace('$', '').replace(',', '').strip()
    
    if s == '' or s == '-':
        return 0
    
    try:
        cents = int(round(float(s) * 100))
        return -cents if negative else cents
    except ValueError:
        return 0


def cents_to_display(cents: int) -> str:
    """Format integer cents as USD display string."""
    if cents < 0:
        return f"-${abs(cents)/100:,.2f}"
    return f"${cents/100:,.2f}"


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


def get_max_transaction_date(direct_costs_df: pd.DataFrame) -> datetime:
    """Get the maximum transaction date from direct costs for as_of_date default."""
    valid_dates = direct_costs_df['date_parsed'].dropna()
    if len(valid_dates) == 0:
        return datetime.now()
    return valid_dates.max().to_pydatetime()


def get_file_hashes() -> Dict[str, str]:
    """Get hashes of all input files for change detection."""
    return {
        'gmp': file_hash(DATA_DIR / "GMP-Amount.xlsx"),
        'budget': file_hash(DATA_DIR / "budget.csv"),
        'direct_costs': file_hash(DATA_DIR / "direct_costs_by_group.csv"),
        'allocations': file_hash(DATA_DIR / "allocations.csv") if (DATA_DIR / "allocations.csv").exists() else ''
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
    def max_transaction_date(self) -> datetime:
        return self._cache.get('max_date', datetime.now())


# Module-level convenience function
def get_data_loader() -> DataLoader:
    return DataLoader()

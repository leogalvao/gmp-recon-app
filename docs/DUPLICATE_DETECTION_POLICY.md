# Duplicate Detection Policy
## GMP Reconciliation App — Direct Cost Deduplication

**Version:** 2.0
**Last Updated:** 2026-01-13
**Status:** Implementation-Ready

---

## Table of Contents
1. [Definitions](#1-definitions)
2. [Normalization Rules](#2-normalization-rules)
3. [Date Field Handling](#3-date-field-handling)
4. [Detection Algorithms](#4-detection-algorithms)
5. [SQL Detection Examples](#5-sql-detection-examples)
6. [Pandas Detection Examples](#6-pandas-detection-examples)
7. [Handling & Resolution](#7-handling--resolution)
8. [Edge Cases & Pitfalls](#8-edge-cases--pitfalls)

---

## 1. Definitions

### 1.1 True Duplicate (Exact Match)

A row is a **True Duplicate** if and only if it is **identical across ALL cells/dimensions** after normalization.

**Formal Definition:**
```
TRUE_DUPLICATE(row_a, row_b) ⟺
  ∀ column ∈ ALL_COLUMNS: normalize(row_a[column]) = normalize(row_b[column])
```

**Matching Columns (Direct Cost Entity):**

| Column Name | Type | Role |
|-------------|------|------|
| `vendor_name` | String | Dimension |
| `description` | String | Dimension |
| `cost_code` | String | Dimension |
| `invoice_number` | String | Dimension |
| `name` | String | Dimension |
| `gross_amount_cents` | Integer | Measure |
| `transaction_date` | Date | Temporal |

**Confidence Score:** `1.0` (100%)

---

### 1.2 Near-Duplicate (Date-Agnostic Match)

A row is a **High-Confidence Near-Duplicate** if all **non-date cells match exactly**, even when date fields differ.

**Formal Definition:**
```
NEAR_DUPLICATE(row_a, row_b) ⟺
  ∀ column ∈ (ALL_COLUMNS - DATE_COLUMNS): normalize(row_a[column]) = normalize(row_b[column])
  ∧ row_a[date_col] ≠ row_b[date_col]  // At least one date differs
```

**Excluded Date Columns:**

| Column Name | Type | Why Excluded |
|-------------|------|--------------|
| `transaction_date` | Date | Primary transaction timestamp |
| `created_at` | DateTime | System metadata |
| `updated_at` | DateTime | System metadata |

**Confidence Score:** `0.95` (95%) — high but not absolute due to date variance.

---

### 1.3 Null, Empty, and Whitespace Treatment

| Value Type | Treatment | Rationale |
|------------|-----------|-----------|
| `NULL` / `None` | Treated as **distinct sentinel** `<NULL>` | Two NULLs match each other but not empty strings |
| Empty string `""` | Normalized to **sentinel** `<EMPTY>` | Distinguishes intentional empty from NULL |
| Whitespace-only `"   "` | Normalized to `<EMPTY>` | Whitespace carries no semantic meaning |
| Mixed whitespace `" abc  "` | Trimmed to `"abc"` | Leading/trailing whitespace ignored |

**Equality Rules:**
```python
NULL  == NULL   → True   (both are <NULL>)
NULL  == ""     → False  (<NULL> ≠ <EMPTY>)
""    == "   "  → True   (both normalize to <EMPTY>)
" A " == "A"    → True   (both normalize to "a")
```

---

### 1.4 Casing Treatment

All string comparisons are **case-insensitive** after normalization:

```
"ACME Corp" == "acme corp" == "Acme CORP" → True
```

---

### 1.5 Numeric Formatting

| Scenario | Treatment |
|----------|-----------|
| Monetary amounts | Compare as **integer cents** (no float drift) |
| Floats with precision variance | Round to **2 decimal places** before integer conversion |
| Accounting negatives `(100.00)` | Convert to `-10000` cents |
| Currency symbols `$1,234.56` | Strip, parse to `123456` cents |

---

## 2. Normalization Rules

All values **MUST** be normalized **before** comparison. Apply these transformations in order:

### 2.1 String Normalization Pipeline

```python
def normalize_string(value: Any) -> str:
    """Normalize string value for comparison."""
    # Step 1: Handle NULL
    if value is None:
        return "<NULL>"

    # Step 2: Convert to string
    s = str(value)

    # Step 3: Unicode normalization (NFC form)
    s = unicodedata.normalize("NFC", s)

    # Step 4: Strip leading/trailing whitespace
    s = s.strip()

    # Step 5: Handle empty/whitespace-only
    if not s or s.isspace():
        return "<EMPTY>"

    # Step 6: Collapse internal whitespace
    s = " ".join(s.split())

    # Step 7: Case-fold to lowercase
    s = s.lower()

    # Step 8: Remove common noise characters (optional, domain-specific)
    s = re.sub(r"[.,;:!?'\"()]", "", s)

    return s
```

### 2.2 Vendor Name Normalization

```python
def normalize_vendor(vendor: str) -> str:
    """Vendor-specific normalization with business suffix removal."""
    s = normalize_string(vendor)
    if s in ("<NULL>", "<EMPTY>"):
        return s

    # Remove common business suffixes
    suffixes = [" inc", " llc", " ltd", " corp", " co", " company", " incorporated"]
    for suffix in suffixes:
        if s.endswith(suffix):
            s = s[:-len(suffix)].rstrip()

    return s
```

### 2.3 Invoice Number Normalization

```python
def normalize_invoice(invoice: str) -> str:
    """Invoice number normalization - alphanumeric only."""
    s = normalize_string(invoice)
    if s in ("<NULL>", "<EMPTY>"):
        return s

    # Keep only alphanumeric characters
    s = re.sub(r"[^a-z0-9]", "", s)

    # Remove leading zeros
    s = s.lstrip("0") or "0"

    return s
```

### 2.4 Monetary Amount Normalization

```python
from decimal import Decimal, ROUND_HALF_UP

def normalize_amount_to_cents(value: Any) -> int:
    """Convert monetary value to integer cents."""
    if value is None:
        return 0

    s = str(value).strip()

    # Handle accounting notation: (100.00) → -100.00
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    # Remove currency symbols and commas
    s = re.sub(r"[$,]", "", s)

    # Parse with Decimal for precision
    try:
        d = Decimal(s)
    except:
        return 0

    # Round to 2 decimal places, convert to cents
    d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(d * 100)
```

---

## 3. Date Field Handling

### 3.1 Identifying Date-Like Columns

Columns are classified as **date-like** if they match any of:

| Pattern | Examples |
|---------|----------|
| Column name contains `date` | `transaction_date`, `created_at`, `as_of_date` |
| Column name contains `_at` suffix | `updated_at`, `resolved_at`, `triggered_at` |
| Column name contains `timestamp` | `import_timestamp` |
| SQLAlchemy type is `Date` or `DateTime` | Introspected from model |

**Date Columns in DirectCostEntity:**
- `transaction_date` (Date) — **Primary business date**
- `created_at` (DateTime) — System metadata
- `updated_at` (DateTime) — System metadata

### 3.2 Date Comparison Strategy

| Scenario | Strategy |
|----------|----------|
| **True Duplicate** | Compare exact date (year-month-day) |
| **Near-Duplicate** | Exclude date columns entirely |
| **DateTime vs Date** | Truncate DateTime to Date before comparison |
| **Time zones** | Convert to UTC, then truncate to date |

### 3.3 Date Normalization

```python
from datetime import date, datetime
import pytz

def normalize_date(value: Any, timezone: str = "UTC") -> str:
    """Normalize date/datetime to YYYY-MM-DD string."""
    if value is None:
        return "<NULL>"

    if isinstance(value, datetime):
        # Convert to UTC if timezone-aware
        if value.tzinfo is not None:
            value = value.astimezone(pytz.UTC)
        value = value.date()

    if isinstance(value, date):
        return value.isoformat()  # "YYYY-MM-DD"

    # Attempt to parse string
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y", "%d-%b-%Y"]:
        try:
            return datetime.strptime(str(value).strip(), fmt).date().isoformat()
        except ValueError:
            continue

    return "<INVALID_DATE>"
```

---

## 4. Detection Algorithms

### 4.1 Plain-English Algorithm

#### True Duplicate Detection

```
FOR EACH unique combination of (normalized_vendor, normalized_invoice,
                                 amount_cents, normalized_date,
                                 normalized_description, normalized_cost_code):
    IF count of rows with this combination > 1:
        MARK all rows in this group as TRUE_DUPLICATES
        ASSIGN same group_id to all rows
        SET confidence = 1.0
        SET method = "exact"
```

#### Near-Duplicate Detection

```
FOR EACH unique combination of (normalized_vendor, normalized_invoice,
                                 amount_cents, normalized_description,
                                 normalized_cost_code):
    # Note: Date columns are EXCLUDED from grouping key

    IF count of rows with this combination > 1:
        IF any two rows have DIFFERENT dates:
            MARK as NEAR_DUPLICATES
            ASSIGN same group_id to all rows
            SET confidence = 0.95
            SET method = "near_duplicate_date_agnostic"
        ELSE:
            # All dates same → this is a TRUE_DUPLICATE, handled above
            SKIP
```

### 4.2 Composite Key Strategy

**True Duplicate Key:**
```
composite_key_exact = hash(
    normalize_vendor([vendor_name]) +
    normalize_invoice([invoice_number]) +
    str([amount_cents]) +
    normalize_date([transaction_date]) +
    normalize_string([description]) +
    normalize_string([cost_code])
)
```

**Near-Duplicate Key:**
```
composite_key_near = hash(
    normalize_vendor([vendor_name]) +
    normalize_invoice([invoice_number]) +
    str([amount_cents]) +
    normalize_string([description]) +
    normalize_string([cost_code])
    # Note: NO date component
)
```

---

## 5. SQL Detection Examples

### 5.1 True Duplicates (Exact Match)

```sql
-- Create normalized view for comparison
WITH normalized_costs AS (
    SELECT
        id,
        uuid,
        -- Normalize vendor
        LOWER(TRIM(COALESCE(vendor_name, '<NULL>'))) AS vendor_norm,
        -- Normalize description
        LOWER(TRIM(COALESCE(description, '<NULL>'))) AS desc_norm,
        -- Normalize cost code
        LOWER(TRIM(COALESCE(cost_code, '<NULL>'))) AS code_norm,
        -- Normalize invoice (alphanumeric only)
        LOWER(REPLACE(REPLACE(REPLACE(
            COALESCE(invoice_number, '<NULL>'), '-', ''), ' ', ''), '.', '')) AS invoice_norm,
        -- Amount already in cents
        gross_amount_cents,
        -- Normalize date
        COALESCE(DATE(transaction_date), '1900-01-01') AS date_norm
    FROM direct_cost_entities
),
-- Find duplicates by exact match on all columns
duplicate_groups AS (
    SELECT
        vendor_norm,
        invoice_norm,
        gross_amount_cents,
        date_norm,
        desc_norm,
        code_norm,
        COUNT(*) AS group_count,
        GROUP_CONCAT(id) AS member_ids
    FROM normalized_costs
    GROUP BY
        vendor_norm,
        invoice_norm,
        gross_amount_cents,
        date_norm,
        desc_norm,
        code_norm
    HAVING COUNT(*) > 1
)
SELECT
    nc.id,
    nc.uuid,
    dg.group_count,
    dg.member_ids,
    'exact' AS method,
    1.0 AS confidence_score
FROM normalized_costs nc
INNER JOIN duplicate_groups dg ON
    nc.vendor_norm = dg.vendor_norm AND
    nc.invoice_norm = dg.invoice_norm AND
    nc.gross_amount_cents = dg.gross_amount_cents AND
    nc.date_norm = dg.date_norm AND
    nc.desc_norm = dg.desc_norm AND
    nc.code_norm = dg.code_norm
ORDER BY dg.member_ids, nc.id;
```

### 5.2 Near-Duplicates (Excluding Date Columns)

```sql
-- Near-duplicates: match on all columns EXCEPT date
WITH normalized_costs AS (
    SELECT
        id,
        uuid,
        LOWER(TRIM(COALESCE(vendor_name, '<NULL>'))) AS vendor_norm,
        LOWER(TRIM(COALESCE(description, '<NULL>'))) AS desc_norm,
        LOWER(TRIM(COALESCE(cost_code, '<NULL>'))) AS code_norm,
        LOWER(REPLACE(REPLACE(REPLACE(
            COALESCE(invoice_number, '<NULL>'), '-', ''), ' ', ''), '.', '')) AS invoice_norm,
        gross_amount_cents,
        COALESCE(DATE(transaction_date), '1900-01-01') AS date_norm
    FROM direct_cost_entities
),
-- Group by all non-date columns
near_dup_groups AS (
    SELECT
        vendor_norm,
        invoice_norm,
        gross_amount_cents,
        desc_norm,
        code_norm,
        COUNT(*) AS group_count,
        COUNT(DISTINCT date_norm) AS unique_dates,
        GROUP_CONCAT(id) AS member_ids
    FROM normalized_costs
    GROUP BY
        vendor_norm,
        invoice_norm,
        gross_amount_cents,
        desc_norm,
        code_norm
    HAVING COUNT(*) > 1
       AND COUNT(DISTINCT date_norm) > 1  -- Must have different dates
)
SELECT
    nc.id,
    nc.uuid,
    ndg.group_count,
    ndg.unique_dates,
    ndg.member_ids,
    'near_duplicate_date_agnostic' AS method,
    0.95 AS confidence_score
FROM normalized_costs nc
INNER JOIN near_dup_groups ndg ON
    nc.vendor_norm = ndg.vendor_norm AND
    nc.invoice_norm = ndg.invoice_norm AND
    nc.gross_amount_cents = ndg.gross_amount_cents AND
    nc.desc_norm = ndg.desc_norm AND
    nc.code_norm = ndg.code_norm
ORDER BY ndg.member_ids, nc.id;
```

### 5.3 Combined Detection Query

```sql
-- Unified duplicate detection with classification
WITH normalized AS (
    SELECT
        id,
        uuid,
        vendor_name,
        LOWER(TRIM(COALESCE(vendor_name, '<NULL>'))) AS vendor_norm,
        LOWER(TRIM(COALESCE(description, '<NULL>'))) AS desc_norm,
        LOWER(TRIM(COALESCE(cost_code, '<NULL>'))) AS code_norm,
        LOWER(REPLACE(REPLACE(COALESCE(invoice_number, '<NULL>'), '-', ''), ' ', '')) AS invoice_norm,
        gross_amount_cents,
        transaction_date,
        COALESCE(DATE(transaction_date), '1900-01-01') AS date_norm
    FROM direct_cost_entities
),
-- Composite key without date
non_date_key AS (
    SELECT
        id,
        vendor_norm || '|' || invoice_norm || '|' ||
        CAST(gross_amount_cents AS TEXT) || '|' ||
        desc_norm || '|' || code_norm AS key_no_date,
        vendor_norm || '|' || invoice_norm || '|' ||
        CAST(gross_amount_cents AS TEXT) || '|' ||
        desc_norm || '|' || code_norm || '|' || date_norm AS key_with_date
    FROM normalized
),
-- Count by each key type
group_stats AS (
    SELECT
        key_no_date,
        key_with_date,
        COUNT(*) OVER (PARTITION BY key_with_date) AS exact_group_size,
        COUNT(*) OVER (PARTITION BY key_no_date) AS near_group_size,
        COUNT(DISTINCT date_norm) OVER (PARTITION BY key_no_date) AS unique_dates_in_group
    FROM non_date_key ndk
    JOIN normalized n ON ndk.id = n.id
)
SELECT
    n.id,
    n.uuid,
    n.vendor_name,
    n.gross_amount_cents,
    n.transaction_date,
    CASE
        WHEN gs.exact_group_size > 1 THEN 'TRUE_DUPLICATE'
        WHEN gs.near_group_size > 1 AND gs.unique_dates_in_group > 1 THEN 'NEAR_DUPLICATE'
        ELSE 'UNIQUE'
    END AS classification,
    CASE
        WHEN gs.exact_group_size > 1 THEN 1.0
        WHEN gs.near_group_size > 1 THEN 0.95
        ELSE NULL
    END AS confidence_score,
    gs.exact_group_size,
    gs.near_group_size
FROM normalized n
JOIN non_date_key ndk ON n.id = ndk.id
JOIN group_stats gs ON ndk.key_with_date = gs.key_with_date
WHERE gs.exact_group_size > 1 OR (gs.near_group_size > 1 AND gs.unique_dates_in_group > 1)
ORDER BY gs.key_no_date, n.id;
```

---

## 6. Pandas Detection Examples

### 6.1 Normalization Functions

```python
import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Dict, Tuple
from decimal import Decimal, ROUND_HALF_UP

# Column configuration
DATE_COLUMNS = ['transaction_date', 'created_at', 'updated_at', 'date_parsed']
DIMENSION_COLUMNS = ['vendor_name', 'description', 'cost_code', 'invoice_number', 'name']
MEASURE_COLUMNS = ['gross_amount_cents', 'amount_cents']

def normalize_string(value) -> str:
    """Normalize string for comparison."""
    if pd.isna(value) or value is None:
        return "<NULL>"
    s = str(value)
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    if not s:
        return "<EMPTY>"
    s = " ".join(s.split())
    s = s.lower()
    return s

def normalize_vendor(vendor) -> str:
    """Vendor-specific normalization."""
    s = normalize_string(vendor)
    if s in ("<NULL>", "<EMPTY>"):
        return s
    suffixes = [" inc", " llc", " ltd", " corp", " co", " company"]
    for suffix in suffixes:
        if s.endswith(suffix):
            s = s[:-len(suffix)].rstrip()
    return s

def normalize_invoice(invoice) -> str:
    """Invoice number normalization."""
    s = normalize_string(invoice)
    if s in ("<NULL>", "<EMPTY>"):
        return s
    s = re.sub(r"[^a-z0-9]", "", s)
    s = s.lstrip("0") or "0"
    return s

def normalize_date(value) -> str:
    """Date normalization to ISO format."""
    if pd.isna(value) or value is None:
        return "<NULL>"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, 'isoformat'):
        return value.isoformat()[:10]
    return str(value)[:10]
```

### 6.2 True Duplicate Detection

```python
def find_true_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Find TRUE duplicates - exact match on ALL columns after normalization.

    Returns:
        - DataFrame with duplicate flags
        - List of duplicate group dictionaries
    """
    df = df.copy()

    # Normalize all columns
    df['_vendor_norm'] = df['vendor_name'].apply(normalize_vendor)
    df['_invoice_norm'] = df.get('invoice_number', df.get('Invoice #', '')).apply(normalize_invoice)
    df['_desc_norm'] = df['description'].apply(normalize_string)
    df['_code_norm'] = df.get('cost_code', df.get('Cost Code', '')).apply(normalize_string)
    df['_date_norm'] = df.get('transaction_date', df.get('date_parsed')).apply(normalize_date)
    df['_amount_cents'] = df.get('gross_amount_cents', df.get('amount_cents', 0)).fillna(0).astype(int)

    # Create composite key for exact matching (includes date)
    df['_exact_key'] = (
        df['_vendor_norm'] + '|' +
        df['_invoice_norm'] + '|' +
        df['_desc_norm'] + '|' +
        df['_code_norm'] + '|' +
        df['_date_norm'] + '|' +
        df['_amount_cents'].astype(str)
    )

    # Find groups with more than one row (true duplicates)
    dup_counts = df.groupby('_exact_key').size()
    dup_keys = dup_counts[dup_counts > 1].index

    # Mark duplicates
    df['is_true_duplicate'] = df['_exact_key'].isin(dup_keys)

    # Build duplicate groups
    duplicates = []
    for group_id, (key, group) in enumerate(df[df['is_true_duplicate']].groupby('_exact_key'), 1):
        row_ids = group.index.tolist()
        for idx in row_ids:
            duplicates.append({
                'row_id': idx,
                'group_id': group_id,
                'method': 'exact',
                'confidence': 1.0,
                'matched_with': [i for i in row_ids if i != idx],
                'group_size': len(row_ids)
            })

    return df, duplicates
```

### 6.3 Near-Duplicate Detection

```python
def find_near_duplicates(df: pd.DataFrame,
                         exclude_exact: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Find NEAR duplicates - match on all columns EXCEPT date columns.

    Args:
        df: Input DataFrame
        exclude_exact: If True, exclude rows already flagged as true duplicates

    Returns:
        - DataFrame with near-duplicate flags
        - List of near-duplicate group dictionaries
    """
    df = df.copy()

    # Normalize non-date columns
    df['_vendor_norm'] = df['vendor_name'].apply(normalize_vendor)
    df['_invoice_norm'] = df.get('invoice_number', df.get('Invoice #', '')).apply(normalize_invoice)
    df['_desc_norm'] = df['description'].apply(normalize_string)
    df['_code_norm'] = df.get('cost_code', df.get('Cost Code', '')).apply(normalize_string)
    df['_amount_cents'] = df.get('gross_amount_cents', df.get('amount_cents', 0)).fillna(0).astype(int)
    df['_date_norm'] = df.get('transaction_date', df.get('date_parsed')).apply(normalize_date)

    # Create composite key WITHOUT date for near-duplicate matching
    df['_near_key'] = (
        df['_vendor_norm'] + '|' +
        df['_invoice_norm'] + '|' +
        df['_desc_norm'] + '|' +
        df['_code_norm'] + '|' +
        df['_amount_cents'].astype(str)
    )

    # Find groups with multiple rows AND multiple distinct dates
    grouped = df.groupby('_near_key').agg({
        '_date_norm': ['count', 'nunique']
    })
    grouped.columns = ['count', 'unique_dates']

    # Near-duplicates: same non-date fields but different dates
    near_dup_keys = grouped[(grouped['count'] > 1) & (grouped['unique_dates'] > 1)].index

    # Mark near-duplicates
    df['is_near_duplicate'] = df['_near_key'].isin(near_dup_keys)

    # Optionally exclude true duplicates
    if exclude_exact and 'is_true_duplicate' in df.columns:
        df.loc[df['is_true_duplicate'], 'is_near_duplicate'] = False

    # Build near-duplicate groups
    duplicates = []
    base_group_id = 5000  # Offset to avoid collision with true duplicate IDs

    for group_id, (key, group) in enumerate(df[df['is_near_duplicate']].groupby('_near_key'), base_group_id):
        row_ids = group.index.tolist()
        dates_in_group = group['_date_norm'].unique().tolist()

        for idx in row_ids:
            duplicates.append({
                'row_id': idx,
                'group_id': group_id,
                'method': 'near_duplicate_date_agnostic',
                'confidence': 0.95,
                'matched_with': [i for i in row_ids if i != idx],
                'group_size': len(row_ids),
                'unique_dates': len(dates_in_group),
                'dates': dates_in_group
            })

    return df, duplicates
```

### 6.4 Combined Detection Pipeline

```python
def detect_all_duplicates(df: pd.DataFrame) -> Dict:
    """
    Complete duplicate detection pipeline.

    Returns dictionary with:
        - df: Annotated DataFrame
        - true_duplicates: List of exact matches
        - near_duplicates: List of date-agnostic matches
        - summary: Statistics
    """
    # Step 1: Find true duplicates
    df, true_dups = find_true_duplicates(df)

    # Step 2: Find near-duplicates (excluding exact matches)
    df, near_dups = find_near_duplicates(df, exclude_exact=True)

    # Step 3: Add unified classification
    df['duplicate_type'] = 'unique'
    df.loc[df['is_true_duplicate'], 'duplicate_type'] = 'true_duplicate'
    df.loc[df['is_near_duplicate'], 'duplicate_type'] = 'near_duplicate'

    # Step 4: Build summary
    summary = {
        'total_rows': len(df),
        'unique_rows': len(df[df['duplicate_type'] == 'unique']),
        'true_duplicate_rows': len(df[df['duplicate_type'] == 'true_duplicate']),
        'near_duplicate_rows': len(df[df['duplicate_type'] == 'near_duplicate']),
        'true_duplicate_groups': len(set(d['group_id'] for d in true_dups)),
        'near_duplicate_groups': len(set(d['group_id'] for d in near_dups)),
        'total_duplicate_amount_cents': df.loc[
            df['duplicate_type'] != 'unique', '_amount_cents'
        ].sum()
    }

    return {
        'df': df,
        'true_duplicates': true_dups,
        'near_duplicates': near_dups,
        'summary': summary
    }


# Usage Example
if __name__ == "__main__":
    # Load direct costs
    df = pd.read_csv("direct_costs.csv")

    # Run detection
    result = detect_all_duplicates(df)

    # Print summary
    print(f"Total rows: {result['summary']['total_rows']}")
    print(f"True duplicates: {result['summary']['true_duplicate_rows']} "
          f"in {result['summary']['true_duplicate_groups']} groups")
    print(f"Near duplicates: {result['summary']['near_duplicate_rows']} "
          f"in {result['summary']['near_duplicate_groups']} groups")

    # Export for review
    dups_df = result['df'][result['df']['duplicate_type'] != 'unique']
    dups_df.to_csv("duplicates_for_review.csv", index=False)
```

---

## 7. Handling & Resolution

### 7.1 True Duplicates — Actions

| Action | Rule | Rationale |
|--------|------|-----------|
| **Keep First** | Retain row with lowest `id` (first ingested) | Deterministic, reproducible |
| **Keep Last** | Retain row with highest `id` (most recent) | May have corrections |
| **Tie-breaker** | If IDs equal, prefer non-NULL description | More complete data |
| **Audit Log** | Always log which rows were excluded | Traceability required |

**Default Policy: Keep First**

```python
def resolve_true_duplicates(df: pd.DataFrame,
                            duplicates: List[Dict],
                            strategy: str = 'keep_first') -> pd.DataFrame:
    """
    Resolve true duplicates by keeping one row per group.

    Args:
        strategy: 'keep_first' (lowest ID) or 'keep_last' (highest ID)
    """
    df = df.copy()
    df['excluded_as_duplicate'] = False

    # Group by group_id
    groups = {}
    for dup in duplicates:
        gid = dup['group_id']
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(dup['row_id'])

    for group_id, row_ids in groups.items():
        sorted_ids = sorted(row_ids)

        if strategy == 'keep_first':
            keep_id = sorted_ids[0]
        else:  # keep_last
            keep_id = sorted_ids[-1]

        exclude_ids = [rid for rid in row_ids if rid != keep_id]
        df.loc[exclude_ids, 'excluded_as_duplicate'] = True

    return df
```

### 7.2 Near-Duplicates — Actions

Near-duplicates require **human review** due to date variance ambiguity.

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| Same vendor/invoice, different dates, same amount | Possible **re-billing** or **date correction** | Flag for review |
| Dates within 7 days | Likely duplicate entry error | Auto-merge candidate (0.85+ confidence) |
| Dates > 30 days apart | Possibly legitimate repeat transaction | Flag only, no auto-merge |

**Scoring Near-Duplicates:**

```python
def score_near_duplicate(group: pd.DataFrame) -> float:
    """
    Score near-duplicate group for merge confidence.

    Returns 0.0 to 1.0 where higher = more likely true duplicate.
    """
    dates = pd.to_datetime(group['_date_norm'], errors='coerce')
    date_span_days = (dates.max() - dates.min()).days

    base_score = 0.95  # Near-duplicate base

    # Penalize large date spans
    if date_span_days > 30:
        base_score -= 0.30
    elif date_span_days > 14:
        base_score -= 0.15
    elif date_span_days > 7:
        base_score -= 0.05

    # Boost if invoice numbers match exactly (not just normalized)
    raw_invoices = group.get('invoice_number', group.get('Invoice #', pd.Series())).unique()
    if len(raw_invoices) == 1:
        base_score += 0.03

    return min(max(base_score, 0.0), 1.0)
```

### 7.3 Grouping Keys for Review UI

```python
def create_review_groups(df: pd.DataFrame,
                         duplicates: List[Dict]) -> List[Dict]:
    """
    Create grouped view for manual review.
    """
    review_groups = []

    groups = {}
    for dup in duplicates:
        gid = dup['group_id']
        if gid not in groups:
            groups[gid] = {
                'group_id': gid,
                'method': dup['method'],
                'confidence': dup['confidence'],
                'rows': []
            }
        groups[gid]['rows'].append(dup['row_id'])

    for gid, group_info in groups.items():
        row_ids = group_info['rows']
        group_df = df.loc[row_ids]

        review_groups.append({
            'group_id': gid,
            'method': group_info['method'],
            'confidence': group_info['confidence'],
            'row_count': len(row_ids),
            'total_amount_cents': group_df['_amount_cents'].sum(),
            'vendors': group_df['vendor_name'].unique().tolist(),
            'dates': group_df['_date_norm'].unique().tolist(),
            'suggested_action': 'keep_first' if group_info['confidence'] >= 0.98 else 'review',
            'rows': group_df.to_dict('records')
        })

    return review_groups
```

### 7.4 Safe Merge Rules

| Field | Can Merge? | Rule |
|-------|-----------|------|
| `vendor_name` | ✅ Yes | Prefer non-NULL, longer value |
| `description` | ✅ Yes | Concatenate unique values |
| `invoice_number` | ⚠️ Caution | Only if normalized forms match |
| `amount_cents` | ❌ No | Must be identical |
| `cost_code` | ⚠️ Caution | Only if normalized forms match |
| `transaction_date` | ⚠️ Caution | For near-dups, prefer earliest |

---

## 8. Edge Cases & Pitfalls

### 8.1 Checklist

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Floating-point drift** | `100.00` vs `99.999999` | Use integer cents throughout |
| **Unicode normalization** | `Café` vs `Cafe´` | Apply NFC normalization |
| **Mixed time zones** | `2026-01-13 23:00 PST` vs `2026-01-14 07:00 UTC` | Convert to UTC before date extraction |
| **Late-arriving data** | Duplicate flagged, original arrives later | Re-run detection on new data load |
| **Case sensitivity** | `ACME` vs `acme` | Case-fold all strings |
| **Trailing zeros** | Invoice `001` vs `1` | Strip leading zeros |
| **NULL vs empty** | NULL treated as matching empty | Use distinct sentinels |
| **Accounting negatives** | `(100.00)` vs `-100.00` | Normalize before comparison |
| **Whitespace variations** | `"  Vendor  Name  "` | Collapse and trim |
| **Business suffix variations** | `Acme Inc` vs `Acme Incorporated` | Remove common suffixes |

### 8.2 Pre-Detection Validation

```python
def validate_before_detection(df: pd.DataFrame) -> List[str]:
    """
    Validate DataFrame before running duplicate detection.
    Returns list of warnings.
    """
    warnings = []

    # Check for missing required columns
    required = ['vendor_name', 'gross_amount_cents']
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")

    # Check for excessive NULLs
    for col in ['vendor_name', 'description']:
        if col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct > 0.5:
                warnings.append(f"Column '{col}' has {null_pct:.1%} NULL values")

    # Check for suspicious amount patterns
    if 'gross_amount_cents' in df.columns:
        zero_count = (df['gross_amount_cents'] == 0).sum()
        if zero_count > len(df) * 0.1:
            warnings.append(f"{zero_count} rows have zero amount")

    # Check date range
    date_col = 'transaction_date' if 'transaction_date' in df.columns else 'date_parsed'
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        date_range = (dates.max() - dates.min()).days
        if date_range > 365 * 3:
            warnings.append(f"Date range spans {date_range} days (>3 years)")

    return warnings
```

### 8.3 Post-Detection Sanity Checks

```python
def sanity_check_results(df: pd.DataFrame,
                         duplicates: List[Dict]) -> List[str]:
    """
    Validate duplicate detection results.
    Returns list of warnings.
    """
    warnings = []

    # Check duplicate ratio
    dup_ratio = len(duplicates) / len(df) if len(df) > 0 else 0
    if dup_ratio > 0.5:
        warnings.append(f"High duplicate ratio: {dup_ratio:.1%}")

    # Check for over-grouping (single group with too many rows)
    group_sizes = {}
    for dup in duplicates:
        gid = dup['group_id']
        group_sizes[gid] = group_sizes.get(gid, 0) + 1

    max_group = max(group_sizes.values()) if group_sizes else 0
    if max_group > 100:
        warnings.append(f"Largest duplicate group has {max_group} rows")

    # Check for NULL-matching false positives
    null_matches = sum(1 for d in duplicates
                       if '<NULL>' in str(d.get('details', {})))
    if null_matches > len(duplicates) * 0.3:
        warnings.append(f"{null_matches} duplicates matched on NULL values")

    return warnings
```

---

## Appendix A: Configuration Constants

```python
# Thresholds and Limits
AUTO_COLLAPSE_THRESHOLD = 0.98      # Auto-exclude duplicates >= this confidence
NEAR_DUPLICATE_BASE_CONFIDENCE = 0.95
DATE_WINDOW_DAYS_FOR_BOOST = 7      # Dates within this window boost confidence
DATE_PENALTY_THRESHOLD_DAYS = 30    # Dates beyond this penalize confidence

# Normalization Settings
REMOVE_BUSINESS_SUFFIXES = True
CASE_INSENSITIVE = True
STRIP_LEADING_ZEROS_INVOICE = True
USE_UNICODE_NFC = True

# Column Mappings (adjust per data source)
COLUMN_ALIASES = {
    'vendor_name': ['Vendor', 'vendor', 'VENDOR', 'Vendor Name'],
    'invoice_number': ['Invoice #', 'invoice', 'InvoiceNumber', 'INV_NUM'],
    'gross_amount_cents': ['amount_cents', 'Amount', 'AMOUNT'],
    'transaction_date': ['date_parsed', 'Date', 'DATE', 'TransactionDate'],
    'description': ['Description', 'desc', 'DESCRIPTION', 'Name'],
    'cost_code': ['Cost Code', 'CostCode', 'COST_CODE', 'code']
}
```

---

## Appendix B: Integration with Existing dedupe.py

This policy extends the existing `app/modules/dedupe.py` implementation:

| Existing Feature | Policy Enhancement |
|-----------------|-------------------|
| `find_exact_duplicates()` | Add description and cost_code to matching key |
| `find_fuzzy_duplicates()` | Add strict near-duplicate mode (no fuzzy, date-only variance) |
| `find_reversals()` | Unchanged — complementary detection method |
| `AUTO_COLLAPSE_THRESHOLD = 0.98` | Align with policy (True Dup = 1.0, Near = 0.95) |

**Recommended Changes to `dedupe.py`:**

1. Add `find_near_duplicates()` function per Section 6.3
2. Update `detect_duplicates()` to include near-duplicate results
3. Add `method='near_duplicate_date_agnostic'` to Duplicate model options
4. Update UI to show near-duplicate groups separately with date variance display

---

*End of Document*

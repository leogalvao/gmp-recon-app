# Performance Anti-Patterns Analysis

This document identifies performance issues in the GMP Recon App codebase, organized by severity and type.

---

## Critical Issues

### 1. Repeated Mapping Inside Loop (N+1 Pattern)

**File:** `app/main.py:3264-3265`

```python
for _, gmp_row in gmp_df.iterrows():
    gmp_division = gmp_row['GMP']
    # ...
    mapped_budget = map_budget_to_gmp(budget_df.copy(), gmp_df)  # REPEATED!
    mapped_direct = map_direct_to_budget(direct_costs_df.copy(), budget_df)  # REPEATED!
```

**Problem:** The `map_budget_to_gmp()` and `map_direct_to_budget()` functions are called inside a loop for EVERY GMP division (~50 iterations). These expensive mapping operations are recomputed 50+ times when they only need to be done once.

**Fix:** Move mapping operations outside the loop:
```python
mapped_budget = map_budget_to_gmp(budget_df.copy(), gmp_df)
mapped_direct = map_direct_to_budget(direct_costs_df.copy(), budget_df)
for _, gmp_row in gmp_df.iterrows():
    division_budget = mapped_budget[mapped_budget['gmp_division'] == gmp_division]
    # ...
```

**Impact:** ~50x performance improvement for batch refresh endpoint.

---

### 2. Triple DataFrame Filtering for Single Value Lookup

**File:** `app/modules/reconciliation.py:531`

```python
budget_desc = budget_df[budget_df['Budget Code'] == budget_code]['Budget Code Description'].iloc[0] \
    if len(budget_df[budget_df['Budget Code'] == budget_code]) > 0 else ''
```

**Problem:** This line filters the DataFrame THREE times for the same condition:
1. `budget_df[budget_df['Budget Code'] == budget_code]` for the `len()` check
2. `budget_df[budget_df['Budget Code'] == budget_code]` for the `iloc[0]` access
3. Each filter is O(n), making this O(3n) per budget code, and O(3n*m) overall when called in a loop

**Fix:** Use dictionary lookup:
```python
# Build once before the loop
budget_desc_map = budget_df.set_index('Budget Code')['Budget Code Description'].to_dict()

# In the loop
budget_desc = budget_desc_map.get(budget_code, '')
```

**Impact:** O(1) lookup instead of O(n) per iteration.

---

### 3. Duplicate DataFrame Iterations

**File:** `app/main.py:817-842`

```python
# First iteration
for _, row in data_loader.budget.iterrows():
    bc = row.get('Budget Code', '')
    budget_options.append({...})

# Second identical iteration
for _, row in data_loader.budget.iterrows():
    bc = row.get('Budget Code', '')
    budget_desc_lookup[bc] = row.get('Budget Code Description', '')
    budget_type_lookup[bc] = row.get('Cost Type', '')
```

**Problem:** Iterating the same DataFrame twice to build similar data structures.

**Fix:** Combine into a single iteration:
```python
budget_options = []
budget_desc_lookup = {}
budget_type_lookup = {}
for _, row in data_loader.budget.iterrows():
    bc = row.get('Budget Code', '')
    if bc:
        budget_options.append({...})
        budget_desc_lookup[bc] = row.get('Budget Code Description', '')
        budget_type_lookup[bc] = row.get('Cost Type', '')
```

---

## High Severity Issues

### 4. Excessive Use of `.iterrows()` (31 instances)

**Locations:**
| File | Line Numbers |
|------|-------------|
| `app/main.py` | 393, 817, 838, 869, 938, 2175, 2209, 2410, 3254, 3349, 3450, 3687, 3827 |
| `app/modules/reconciliation.py` | 167, 235, 410, 535 |
| `app/modules/mapping.py` | 95, 170, 190, 285, 352 |
| `app/modules/suggestion_engine.py` | 691, 713 |
| `app/modules/etl.py` | 551, 1077, 1136 |
| `app/modules/dedupe.py` | 42 |
| `src/schedule/parser.py` | 361 |
| `src/schedule/cost_allocator.py` | 92 |
| `src/features/schedule_driven_features.py` | 129, 331 |

**Problem:** `.iterrows()` is notoriously slow in pandas because it:
- Creates a new Series object for each row
- Loses dtype optimization
- Can be 100-1000x slower than vectorized operations

**Example (app/main.py:869-890):**
```python
all_budget_items = []
for _, row in budget_df.iterrows():
    item = {
        'Budget Code': bc,
        'Budget Code Description': row.get('Budget Code Description', ''),
        # ...
    }
    all_budget_items.append(item)
```

**Fix options:**
1. Use `.to_dict('records')` for direct conversion
2. Use vectorized operations with `.apply()` for transformations
3. Use `itertuples()` which is 10-100x faster than `iterrows()`

```python
# Option 1: Direct conversion
all_budget_items = budget_df.to_dict('records')

# Option 2: Use itertuples() if iteration is necessary
for row in budget_df.itertuples():
    item = {'Budget Code': row.Budget_Code, ...}
```

---

### 5. Row-wise Lambda Application Instead of Vectorization

**File:** `app/main.py:276-281`

```python
direct_costs_df = direct_costs_df[
    direct_costs_df.apply(
        lambda r: (r.get('Cost Code', ''), r.get('Name', '')) in side_direct_keys,
        axis=1
    )
]
```

**Problem:** Using `apply()` with `axis=1` iterates over every row, losing pandas vectorization benefits.

**Fix:** Use vectorized set membership:
```python
# Create tuple keys vectorized
keys = list(zip(direct_costs_df['Cost Code'], direct_costs_df['Name']))
mask = pd.Series(keys).isin(side_direct_keys)
direct_costs_df = direct_costs_df[mask.values]
```

Or use merge for better performance:
```python
side_df = pd.DataFrame(list(side_direct_keys), columns=['Cost Code', 'Name'])
direct_costs_df = direct_costs_df.merge(side_df, on=['Cost Code', 'Name'], how='inner')
```

---

## Medium Severity Issues

### 6. Repeated Database Queries for Same Configuration

**File:** `app/main.py` (lines 431, 488, 638, 783)

```python
# Same query repeated 4 times across different endpoints
side_configs = db.query(SideConfiguration).filter(SideConfiguration.is_active == True).all()
```

**Problem:** Static configuration data is queried from the database 4+ times in different endpoints, even though it rarely changes.

**Fix:** Use caching with TTL:
```python
from functools import lru_cache
from datetime import datetime, timedelta

_side_config_cache = None
_side_config_cache_time = None
CACHE_TTL = timedelta(minutes=5)

def get_active_side_configs(db: Session):
    global _side_config_cache, _side_config_cache_time
    now = datetime.now()
    if _side_config_cache is None or (now - _side_config_cache_time) > CACHE_TTL:
        _side_config_cache = db.query(SideConfiguration).filter(
            SideConfiguration.is_active == True
        ).all()
        _side_config_cache_time = now
    return _side_config_cache
```

---

### 7. Loading All Records Before Filtering

**File:** `app/main.py:294-303`

```python
breakdown_records = db.query(GMPBudgetBreakdown).all()  # Loads ALL records
breakdown_df = None
if breakdown_records:
    breakdown_df = pd.DataFrame([{...} for b in breakdown_records if b.gmp_division])
```

**Problem:** Loads all records from database into memory, then filters in Python. Better to filter at the database level.

**Fix:**
```python
breakdown_records = db.query(GMPBudgetBreakdown).filter(
    GMPBudgetBreakdown.gmp_division.isnot(None)
).all()
```

---

### 8. Nested Loop in Reconciliation Aggregation

**File:** `app/modules/reconciliation.py:167-199`

```python
agg_totals = merged.groupby('gmp_division').agg({...}).reset_index()

for _, row in agg_totals.iterrows():
    gmp_div = row['gmp_division']
    # ...
    if use_breakdown_allocations and breakdown_df is not None:
        east, west = allocate_amount_east_west(total, gmp_div, breakdown_df)
    else:
        div_data = merged[merged['gmp_division'] == gmp_div]  # Re-filters merged!
```

**Problem:** After groupby aggregation, the code filters the original `merged` DataFrame again for each GMP division, creating an O(n*m) operation.

**Fix:** Pre-compute all group data or use a dictionary of grouped DataFrames:
```python
grouped = {name: group for name, group in merged.groupby('gmp_division')}
for gmp_div, group_data in grouped.items():
    # Use group_data directly instead of filtering
```

---

## Low Severity Issues

### 9. Sorting After Building Full Lists

**File:** `app/main.py:893-897`

```python
all_unmapped_budget = [b for b in all_budget_items if not b['is_mapped']]
all_unmapped_budget.sort(key=lambda x: (-float(x.get('suggestion_confidence', 0) or 0), ...))
```

**Improvement:** If only showing top N items, use `heapq.nlargest()` for O(n log k) instead of O(n log n).

---

### 10. Multiple Sequential DataFrame Operations in ETL

**File:** `app/modules/etl.py:305-313`

```python
for col in ['Current Budget', 'Committed Costs', 'Direct Cost Tool']:
    df[f'{col.lower().replace(" ", "_")}_cents'] = df[col].apply(parse_money_to_cents)

df['base_code'] = df['Budget Code'].apply(normalize_code)
df['division_key'] = df['Cost Code Tier 2'].apply(extract_division_key)
```

**Improvement:** Batch column operations where possible to reduce DataFrame overhead.

---

## Summary

| Category | Count | Est. Performance Impact |
|----------|-------|------------------------|
| Mapping in loops (Critical) | 1 | 50x slowdown |
| Triple filtering (Critical) | 1 | 3x slowdown per iteration |
| Duplicate iterations (Critical) | 1 | 2x slowdown |
| `.iterrows()` usage | 31 | 10-100x per instance |
| Row-wise apply | 2 | 10-50x per instance |
| Repeated DB queries | 4 | Network latency * 4 |
| Load all then filter | 2 | Memory + transfer waste |
| Nested aggregation | 2 | O(n*m) instead of O(n) |

---

## Recommended Fix Priority

1. **Immediate (Critical):** Fix batch refresh loop (`main.py:3264`) - largest single impact
2. **Immediate (Critical):** Fix triple filtering in reconciliation drill-down (`reconciliation.py:531`)
3. **Immediate (Critical):** Combine duplicate iterations (`main.py:817-842`)
4. **High Priority:** Replace `.iterrows()` with vectorized operations in hot paths
5. **High Priority:** Add caching for side configuration queries
6. **Medium Priority:** Fix remaining `.iterrows()` instances
7. **Medium Priority:** Move database filtering to query level
8. **Low Priority:** Optimize sorting and minor ETL operations

---

## Estimated Impact

Addressing the critical and high-priority issues could yield:
- **Batch refresh endpoint:** 50-100x faster
- **Drill-down endpoints:** 10-20x faster
- **Main reconciliation page:** 5-10x faster
- **Overall memory usage:** 30-50% reduction

# Dependency Audit Report

**Date:** 2026-01-17 (Updated)
**Previous Audit:** 2026-01-09
**Audited by:** Claude Code

## Executive Summary

This audit identified **7 security vulnerabilities** in 3 packages, **1 unused dependency** (removed), and significant **optimization opportunities** to reduce the install footprint by ~80% for users who don't need ML features.

**Status:** Security constraints have been added to `requirements.txt` to enforce minimum secure versions.

## Security Vulnerabilities (Critical)

| Package | System Version | Fixed Version | CVEs | Status |
|---------|----------------|---------------|------|--------|
| `cryptography` | 41.0.7 | **≥43.0.1** | CVE-2023-50782, CVE-2024-0727, PYSEC-2024-225, GHSA-h4gh-qq45-vh27 | ✅ Pinned |
| `setuptools` | 68.1.2 | **≥78.1.1** | CVE-2024-6345 (RCE), PYSEC-2025-49 (path traversal) | ✅ Pinned |
| `pip` | 24.0 | **≥25.3** | CVE-2025-8869 (tar symlink attack) | ⚠️ Manual upgrade needed |
| `urllib3` | 2.6.3 | **≥2.6.3** | CVE-2026-21441 (decompression bomb) | ✅ Already fixed |

### Recommended Actions
1. ✅ `cryptography>=43.0.1` - Added to requirements.txt
2. ✅ `setuptools>=78.1.1` - Added to requirements.txt
3. ⚠️ `pip>=25.3` - Run `pip install --upgrade pip` before installing other packages
4. ✅ `urllib3>=2.6.3` - Added to requirements.txt

## Unused Dependencies

| Package | Status | Evidence |
|---------|--------|----------|
| `mlflow>=2.8.0` | ✅ **REMOVED** | No imports found (`import mlflow`, `from mlflow`, `mlflow.`) |

**Status:** `mlflow` has been removed from requirements.txt. Saved ~150MB+ of transitive dependencies.

## Dependency Bloat Analysis

### Heavy ML Dependencies

The current `requirements.txt` includes both **TensorFlow** (~500MB+) and **PyTorch** (~2GB) which is excessive for most use cases.

| Package | Size | Usage | Recommendation |
|---------|------|-------|----------------|
| `tensorflow>=2.15.0` | ~500MB | Used in `app/forecasting/` (LSTM, Transformer models) | Move to `[forecasting]` extras |
| `torch>=2.0.0` | ~2GB | Optional in `app/modules/ml.py` (MLP fallback) | Move to `[ml]` extras |

### Usage Analysis

| Dependency | Files Using | Purpose | Required |
|------------|-------------|---------|----------|
| `pandas` | 15 files | Core data processing | ✅ Yes |
| `numpy` | 16 files | Numerical operations | ✅ Yes |
| `scikit-learn` | 2 files | Linear regression, metrics | ✅ Yes |
| `fastapi` | Core app | Web framework | ✅ Yes |
| `sqlalchemy` | 35 files | Database ORM | ✅ Yes |
| `rapidfuzz` | 5 files | String matching | ✅ Yes |
| `apscheduler` | 1 file | Scheduled jobs | ✅ Yes |
| `click` | 1 file | CLI only | ⚠️ CLI only |
| `tensorflow` | 7 files | Advanced forecasting | ⚠️ Optional |
| `torch` | 1 file (conditional) | Optional MLP | ⚠️ Optional |
| `mlflow` | 0 files | Not used | ❌ Remove |
| `httpx` | 0 files (test only) | Testing FastAPI | ⚠️ Test only |

## Recommended requirements.txt Structure

### Option A: Single File with Comments (Simple)

```txt
# Core Dependencies (required)
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
jinja2>=3.1.2
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
click>=8.0.0
sqlalchemy>=2.0.0
apscheduler>=3.10.0
rapidfuzz>=3.0.0
python-dateutil>=2.8.0
python-multipart>=0.0.6
pydantic>=2.0.0
pyyaml>=6.0.0

# Testing
pytest>=7.0.0
httpx>=0.25.0

# Optional: Advanced Forecasting (install with: pip install tensorflow>=2.15.0)
# tensorflow>=2.15.0

# Optional: PyTorch MLP (install with: pip install torch>=2.0.0)
# torch>=2.0.0
```

### Option B: Extras in pyproject.toml (Recommended)

```toml
[project.optional-dependencies]
forecasting = ["tensorflow>=2.15.0"]
ml = ["torch>=2.0.0"]
dev = ["pytest>=7.0.0", "httpx>=0.25.0"]
all = ["tensorflow>=2.15.0", "torch>=2.0.0"]
```

This allows:
- `pip install .` - Core only (~200MB)
- `pip install .[forecasting]` - With TensorFlow (~700MB)
- `pip install .[ml]` - With PyTorch (~2.2GB)
- `pip install .[all]` - Everything (~2.7GB)

## Install Size Comparison

| Profile | Size | Use Case |
|---------|------|----------|
| Core only | ~200MB | Basic reconciliation, no ML |
| + forecasting | ~700MB | LSTM/Transformer forecasting |
| + ml | ~2.2GB | Optional MLP model |
| Full install | ~2.7GB | All features |

## Additional Recommendations

### 1. Pin Security-Critical Packages

Add minimum versions for security fixes:

```txt
# Security minimums
cryptography>=43.0.1
urllib3>=2.6.3
```

### 2. Consider Lighter Alternatives

- **TensorFlow vs PyTorch:** Choose one. If TensorFlow is primary (for LSTM), remove PyTorch since sklearn's LinearRegression is the fallback.
- **keras-only:** For simpler deployments, consider `tf-keras` standalone.

### 3. Add Version Constraints File

Create `constraints.txt` for reproducible builds:

```txt
# Pin transitive dependencies with security fixes
cryptography>=43.0.1
setuptools>=78.1.1
urllib3>=2.6.3
```

### 4. Test Requirements Separation

Move test-only dependencies to `requirements-dev.txt`:

```txt
pytest>=7.0.0
httpx>=0.25.0
```

## Action Items

1. ✅ **Immediate:** Remove `mlflow>=2.8.0` (unused) - **DONE**
2. ✅ **Immediate:** Add security minimum versions - **DONE** (cryptography, setuptools, urllib3 pinned)
3. ⚠️ **Short-term:** Move `tensorflow` and `torch` to optional extras - `torch` commented, `tensorflow` documented
4. 📋 **Short-term:** Create separate `requirements-dev.txt` for test deps
5. 📋 **Medium-term:** Consider migrating to `pyproject.toml` for modern dependency management

## Installation Notes

Before installing dependencies, upgrade pip to fix CVE-2025-8869:
```bash
pip install --upgrade pip>=25.3
pip install -r requirements.txt
```

For minimal install (no advanced ML features), comment out `tensorflow>=2.15.0` in requirements.txt to save ~500MB.

## Files Modified

- `requirements.txt` - Updated with security fixes, explicit version constraints, and removed unused deps
- `DEPENDENCY_AUDIT.md` - Updated with audit findings and status

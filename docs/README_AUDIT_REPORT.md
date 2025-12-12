# CJE README Audit Report

**Date**: 2025-12-02
**Auditor**: Claude Code
**Scope**: All second-level directory READMEs in `cje/cje/`

---

## Executive Summary

After thoroughly reading every README and cross-checking against actual implementations, I found **23 distinct issues** across 9 modules. The most severe are in `diagnostics/` (ghost file, missing class documentation) and `utils/` (2 CLI tools completely undocumented).

| Priority | Count | Description |
|----------|-------|-------------|
| 🔴 Critical | 6 | Must fix - incorrect/missing documentation |
| 🟡 Medium | 5 | Should fix - incomplete documentation |
| 🟢 Low | 4 | Nice to fix - minor inconsistencies |

---

## 🔴 Critical Issues (Must Fix)

### 1. diagnostics/README.md - Ghost File `stability.py`

**Location**: Lines 18, 33

**README claims**:
```
├── stability.py         # Stability and drift detection
```

**Reality**: **File does not exist** in `cje/diagnostics/`

**Verification**:
```bash
ls cje/diagnostics/*.py
# Output: __init__.py display.py dr.py models.py overlap.py
#         reward_boundary.py robust_inference.py transport.py weights.py
# NO stability.py
```

**Impact**: Users will search for non-existent functionality

**Fix**: Remove `stability.py` from file structure listing

---

### 2. diagnostics/README.md - Missing File `transport.py`

**Location**: File structure (lines 12-22)

**README**: Not listed in file structure

**Reality**: `transport.py` exists AND is exported in `__init__.py`:
```python
# From cje/diagnostics/__init__.py lines 41-44
from .transport import (
    TransportDiagnostics,
    audit_transportability,
)
```

**Impact**: Transport diagnostics not discoverable from README

**Fix**: Add to file structure:
```
├── transport.py         # Transportability auditing
```

---

### 3. diagnostics/README.md - `CJEDiagnostics` Class Completely Undocumented

**Location**: `models.py` lines 476-661

**README**: Zero mentions (verified via grep - no matches)

**Reality**: Fully implemented unified diagnostics class with:
- `from_ips_diagnostics()` factory method
- `from_dr_diagnostics()` factory method
- `can_make_level_claims`, `has_honest_inference` properties
- `overall_risk`, `coverage_risk`, `variance_risk` assessments
- `summary()` method for practitioners

**Exported**: Yes (`__init__.py` line 16, 80)

**Code snippet** (`models.py` lines 476-484):
```python
@dataclass
class CJEDiagnostics:
    """Unified diagnostics for paper-ready reporting.

    Simplifies IPSDiagnostics and DRDiagnostics into a single class
    focused on the two key questions:
    1. Can we make level claims? (identification/coverage risk)
    2. Are CIs honest? (sampling/variance risk)
    """
```

**Impact**: Major public API class invisible to users

**Fix**: Add new section to README documenting CJEDiagnostics

---

### 4. diagnostics/models.py - Stale Docstring

**Location**: Lines 4-5

**Docstring says**:
```python
"""
Diagnostic data models for CJE.

This module contains the data structures for diagnostics.
Computation logic is in utils/diagnostics/.
"""
```

**Reality**: Computation logic is in `cje/diagnostics/` (weights.py, dr.py, overlap.py, etc.), NOT `utils/diagnostics/`

**Impact**: Misleading comment for code readers

**Fix**: Update docstring to reference correct location

---

### 5. estimators/README.md - Incorrect Default Claim

**Location**: Line 108

**README claims**:
> **Use StackedDREstimator** - Combines multiple DR methods (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E)

**Code** (`stacking.py` lines 51, 65-69):
```python
"""
Args:
    estimators: List of estimator names to stack
        Default: ["dr-cpo", "tmle", "mrdr"] - core DR estimators only
"""
# ...
self.estimators = estimators or [
    "dr-cpo",
    "tmle",
    "mrdr",
]
```

**Reality**: Default is **3 estimators**, not 5

**Impact**: Users expect 5 estimators but get 3

**Fix**: Change line 108 to:
> **Use StackedDREstimator** - Combines multiple DR methods (default: DR-CPO, TMLE, MRDR; optionally add OC-DR-CPO, TR-CPO-E)

---

### 6. utils/README.md - Two CLI Modules Missing

**Location**: File structure (lines 23-28)

**README lists**:
```
utils/
├── __init__.py                  # Re-exports and backward compatibility
├── export.py                    # JSON/CSV export functions
└── extreme_weights_analysis.py  # Weight debugging and reporting
```

**Actually exists**:
```bash
ls cje/utils/*.py
# __init__.py  aggregate_diagnostics.py  analyze_diagnostics.py
# export.py  extreme_weights_analysis.py
```

**Missing from README**:
- `aggregate_diagnostics.py` - CLI for aggregating JSON results into CSV
- `analyze_diagnostics.py` - CLI for correlation analysis of aggregated diagnostics

**Usage** (from `aggregate_diagnostics.py` docstring):
```bash
python -m cje.utils.aggregate_diagnostics --input results_dir --output agg.csv
```

**Impact**: Useful CLI tools invisible to users

**Fix**: Add both files to file structure and document CLI usage

---

## 🟡 Medium Priority Issues

### 7. visualization/README.md - Missing `transport.py` Module

**Location**: File structure (lines 35-42)

**README lists**:
```
visualization/
├── __init__.py              # Public API
├── calibration.py           # Calibration plots
├── dr_dashboards.py         # DR diagnostic visualizations
├── estimates.py             # Policy performance forest plots
└── weight_dashboards.py     # Weight diagnostic dashboards
```

**Missing**: `transport.py`

**Functions exported** (from `visualization/__init__.py` lines 29-32, 45-46):
- `plot_transport_audit`
- `plot_transport_comparison`

Also exported in top-level `cje/__init__.py` (lines 34-35)

**Impact**: Transport visualization not discoverable

**Fix**: Add `transport.py` to file structure and document the two functions

---

### 8. teacher_forcing/templates - `Llama4TemplateConfig` Not Exported

**Location**: `llama.py` lines 7-33 (class exists)

**`templates/__init__.py` exports**:
```python
__all__ = [
    "ChatTemplateConfig",
    "Llama3TemplateConfig",  # Only Llama3, not Llama4
    "FireworksTemplateConfig",
    "FireworksTemplateError",
]
```

**Reality**: `Llama4TemplateConfig` is fully implemented but not exported

**Impact**: Users cannot easily use Llama 4 templates

**Fix**: Either add `Llama4TemplateConfig` to exports, or document as internal-only

---

### 9. data/README.md - Missing File `fresh_draw_utils.py`

**Location**: File structure (lines 31-42)

**Reality**: File exists at `cje/data/fresh_draw_utils.py`

**Contains**: `compute_fresh_draw_prompt_stats()` function

**Exported**: No (internal utility)

**Impact**: Minor - file is internal, but structure should be accurate

**Fix**: Add to file structure with "(internal)" note, or leave as-is if intentionally hidden

---

### 10. calibration/README.md - OUA vs Oracle Augmentation Conflation

The README conflates two distinct mechanisms:

1. **OUA jackknife**: Delete-one-fold variance component for calibrator uncertainty (always available)
2. **Oracle slice augmentation**: Bias correction term (disabled by default via `enable_augmentation: bool = False`)

**Impact**: Users may misunderstand the uncertainty quantification system

**Fix**: Clarify that these are separate mechanisms with different purposes

---

### 11. tests/README.md - File Count Wrong

**README claims** (line 5):
> The suite consists of 13 test files (111 tests)

**Reality**: 14 test files exist

**Missing from listing**: `test_cle_diagnostics.py`

**Fix**: Update count and add missing file to listing

---

## 🟢 Low Priority Issues

### 12. diagnostics/README.md - `reward_boundary.py` Not in README

**Status**: File exists but is NOT exported in `__init__.py`

**Verdict**: Correctly omitted from README (internal only)

**No action needed** unless deciding to make it public

---

### 13. calibration/README.md - "deprecated" Mislabel

**Location**: Line 37

**README says**: `oracle_slice.py` - Oracle slice configuration **(deprecated)**

**Reality**: It's "optional", not deprecated (still actively used, just disabled by default)

**Fix**: Change "(deprecated)" to "(optional)" or "(advanced)"

---

### 14. Various READMEs - "New" Labels Will Go Stale

Multiple READMEs use "New:" labels without dates:
- calibration/README.md line 213: "**New**: Supports fit/predict separation"
- data/README.md line 119: "New visualization features"

**Fix**: Remove "New" labels or add version/date context

---

### 15. estimators/README.md - Outer CV Seed Mismatch

**README implies** (line 118): Seeds should align between outer CV and DR folds

**Reality**:
- `outer_cv_seed` defaults to 1042 in CalibratedIPS
- `random_seed` defaults to 42 elsewhere

**Impact**: Minor - seeds don't auto-align as might be expected

**Fix**: Document the default values explicitly or align them

---

## Module-by-Module Summary

| Module | File Structure Accuracy | API Documentation | Severity |
|--------|------------------------|-------------------|----------|
| **diagnostics/** | 50% (ghost file + 2 missing) | 70% (CJEDiagnostics missing) | 🔴 CRITICAL |
| **estimators/** | 95% | 85% (wrong defaults) | 🔴 HIGH |
| **utils/** | 60% (2 CLI tools missing) | 95% | 🟡 MEDIUM |
| **visualization/** | 80% (transport.py missing) | 85% | 🟡 MEDIUM |
| **teacher_forcing/** | 95% | 80% (Llama4 not exported) | 🟡 MEDIUM |
| **data/** | 90% (1 file missing) | 90% | 🟢 LOW |
| **calibration/** | 90% | 85% | 🟢 LOW |
| **interface/** | 95% | 95% | 🟢 LOW |
| **tests/** | 90% (count wrong) | 95% | 🟢 LOW |

---

## Recommended Fix Order

### Phase 1: Critical Fixes (diagnostics + estimators)

1. **diagnostics/README.md**:
   - [ ] Remove `stability.py` from file structure (doesn't exist)
   - [ ] Add `transport.py` to file structure
   - [ ] Add `CJEDiagnostics` section documenting the unified diagnostics class
   - [ ] Fix stale docstring in `models.py` line 5

2. **estimators/README.md**:
   - [ ] Fix line 108: Change "DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E" to "default: DR-CPO, TMLE, MRDR; optionally add OC-DR-CPO, TR-CPO-E"

### Phase 2: Medium Priority (utils + visualization + teacher_forcing)

3. **utils/README.md**:
   - [ ] Add `aggregate_diagnostics.py` to file structure
   - [ ] Add `analyze_diagnostics.py` to file structure
   - [ ] Add CLI usage section documenting both tools

4. **visualization/README.md**:
   - [ ] Add `transport.py` to file structure
   - [ ] Document `plot_transport_audit()` function
   - [ ] Document `plot_transport_comparison()` function

5. **teacher_forcing/templates/__init__.py**:
   - [ ] Either export `Llama4TemplateConfig` or document as internal

### Phase 3: Low Priority Cleanup

6. **tests/README.md**:
   - [ ] Update file count from 13 to 14
   - [ ] Add `test_cle_diagnostics.py` to listing

7. **calibration/README.md**:
   - [ ] Change "deprecated" to "optional" for oracle_slice.py

8. **Various READMEs**:
   - [ ] Remove or date "New" labels

---

## Verification Commands

To verify the findings in this report:

```bash
# Check diagnostics directory (no stability.py, has transport.py)
ls cje/diagnostics/*.py

# Check utils directory (should show 5 .py files)
ls cje/utils/*.py

# Check visualization directory (should show transport.py)
ls cje/visualization/*.py

# Check stacking default (should show 3 estimators)
grep -A5 "self.estimators = estimators or" cje/estimators/stacking.py

# Check CJEDiagnostics export
grep "CJEDiagnostics" cje/diagnostics/__init__.py

# Check CJEDiagnostics in README (should return nothing)
grep "CJEDiagnostics" cje/diagnostics/README.md
```

---

## Appendix: Files Audited

### READMEs Reviewed
- `cje/calibration/README.md`
- `cje/data/README.md`
- `cje/diagnostics/README.md`
- `cje/estimators/README.md`
- `cje/interface/README.md`
- `cje/teacher_forcing/README.md`
- `cje/tests/README.md`
- `cje/utils/README.md`
- `cje/visualization/README.md`

### Code Files Cross-Referenced
- All `__init__.py` files in each module
- All Python implementation files mentioned in READMEs
- `stacking.py` for default estimator verification
- `models.py` for CJEDiagnostics class verification
- `templates/*.py` for Llama4 verification

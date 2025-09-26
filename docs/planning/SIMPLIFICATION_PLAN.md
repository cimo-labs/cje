# CJE Simplification Plan

## Executive Summary

This document outlines a comprehensive plan to simplify CJE's API and structure before public release. Since we have no existing users, we can make breaking changes to achieve optimal simplicity and usability.

## Current State Analysis

### Problems Identified

1. **API Surface Too Large**
   - 50+ exports from `cje.__init__.py`
   - Users don't know where to start
   - Too many ways to do the same thing

2. **Estimator Proliferation**
   - 7 different estimators (RawIPS, CalibratedIPS, DRCPOEstimator, MRDREstimator, TMLEEstimator, MRDRTMLEEstimator)
   - Users don't know which to choose
   - Most users only need 1-2

3. **Configuration Complexity**
   - Too many parameters without guidance
   - No presets or smart defaults
   - Expert knowledge required for basic usage

4. **Mixed Audiences**
   - Research code mixed with production code
   - No clear separation of simple vs advanced
   - Documentation treats everything equally

### Strengths to Preserve

1. **Core Functionality**
   - `analyze_dataset()` provides one-line interface
   - Automatic oracle augmentation works well
   - Module structure is logical

2. **Technical Excellence**
   - Solid theoretical foundation
   - Comprehensive diagnostics
   - Well-tested core algorithms

## Key Insight: No Users = Maximum Freedom

Since CJE has no existing users, we have a unique opportunity to:
- Make bold simplifications without migration concerns
- Remove complexity rather than hide it
- Be opinionated about the "right" way
- Ship a clean 1.0 API from the start

This freedom should be used aggressively to create the simplest possible API.

## Proposed Architecture

### Three-Layer API Design

```
┌─────────────────────────────────────────┐
│         Layer 1: Simple API             │
│         cje.__init__.py                 │
│    (analyze_dataset + minimal exports)  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Layer 2: Advanced API           │
│         cje.advanced.*                  │
│    (CalibratedIPS, PrecomputedSampler)  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Layer 3: Research API           │
│         cje.research.*                  │
│    (MRDR, TMLE, experimental features)  │
└─────────────────────────────────────────┘
```

### Design Principles

1. **One Obvious Way**: Each layer has one clear way to accomplish tasks
2. **Progressive Disclosure**: Complexity revealed only when needed
3. **Smart Defaults**: The tool makes good choices for users
4. **Fail Loudly**: Better to error than silently give bad results

## Before/After Comparison

### User Experience: Before
```python
# Confusing: Which estimator? What parameters?
from cje import (
    PrecomputedSampler, CalibratedIPS, RawIPS,
    calibrate_dataset, load_dataset_from_jsonl,
    compute_teacher_forced_logprob, CalibrationResult,
    # ... many more imports
)

dataset = load_dataset_from_jsonl("data.jsonl")
calibrated = calibrate_dataset(dataset, judge_field="judge_score")
sampler = PrecomputedSampler(calibrated)
estimator = CalibratedIPS(sampler, var_cap=0.5)  # What's var_cap?
results = estimator.fit_and_estimate()
```

### User Experience: After
```python
# Simple: One function, smart defaults
from cje import analyze_dataset

results = analyze_dataset("data.jsonl")
print(f"Best policy: {results.best_policy}")
```

### Power User: Before
```python
# Same as regular user - no clear progression
```

### Power User: After
```python
# Clear advanced path when needed
from cje.advanced import CalibratedIPS, PrecomputedSampler

# Manual control available but not required
dataset = load_dataset("data.jsonl")
sampler = PrecomputedSampler(dataset)
estimator = CalibratedIPS(sampler, preset="conservative")
results = estimator.fit_and_estimate()
```

## Implementation Plan

### Phase 1: API Simplification

#### 1.1 Reduce Top-Level Exports

**Current** (50+ exports):
```python
__all__ = [
    "BaseCJEEstimator", "CalibratedIPS", "RawIPS", 
    "PrecomputedSampler", "Sample", "Dataset",
    "compute_teacher_forced_logprob", "calibrate_dataset",
    # ... 40+ more
]
```

**Proposed** (5 exports):
```python
__all__ = [
    # Primary interface
    "analyze_dataset",
    
    # Data inspection
    "load_dataset_from_jsonl",
    
    # Results type (for type hints)
    "EstimationResult",
    
    # Version
    "__version__",
    
    # Advanced modules (namespaces only)
    "advanced",
]
```

#### 1.2 Simplify Estimator Names and Choices

**Current**:
- `calibrated-ips` - Calibrated importance sampling
- `raw-ips` - Raw importance sampling  
- `dr-cpo` - Doubly robust CPO
- `mrdr` - Multiply robust DR
- `tmle` - Targeted maximum likelihood
- `mrdr-tmle` - MRDR + TMLE

**Proposed**:
```python
# In analyze_dataset()
estimator: Literal["auto", "ips", "dr"] = "auto"

# "auto" mode logic:
if has_fresh_draws and ess < 0.1:
    use "dr"
else:
    use "ips"  # (always calibrated by default)
```

#### 1.3 Configuration Presets

**Current**:
```python
CalibratedIPS(sampler, var_cap=0.5, ess_floor=0.01, 
              include_baseline=False, ...)
```

**Proposed**:
```python
# In analyze_dataset()
analyze_dataset("data.jsonl", 
    robustness="auto",  # auto/low/medium/high
    estimator="auto"    # auto/ips/dr
)

# Presets internally map to:
ROBUSTNESS_PRESETS = {
    "low": {"var_cap": 0.7, "ess_floor": 0.001},
    "medium": {"var_cap": 0.5, "ess_floor": 0.01},  # default
    "high": {"var_cap": 0.3, "ess_floor": 0.05},
    "auto": None,  # Choose based on diagnostics
}
```

### Phase 2: Module Reorganization

#### 2.1 Create `cje.advanced` Module

```python
# cje/advanced/__init__.py
"""Advanced API for power users who need manual control."""

from ..estimators import CalibratedIPS
from ..data import PrecomputedSampler
from ..calibration import calibrate_dataset

__all__ = [
    "CalibratedIPS",
    "PrecomputedSampler", 
    "calibrate_dataset",
]
```

#### 2.2 Create `cje.research` Module

```python
# cje/research/__init__.py
"""Research API with experimental estimators and features."""

from ..estimators import (
    RawIPS,  # For comparison only
    MRDREstimator,
    TMLEEstimator,
    MRDRTMLEEstimator,
)

__all__ = [
    "RawIPS",
    "MRDREstimator",
    "TMLEEstimator", 
    "MRDRTMLEEstimator",
]
```

#### 2.3 Hide Internal Modules

Move these to `cje._internal`:
- `teacher_forcing` (users shouldn't need this directly)
- `utils` (internal utilities)
- Complex calibration functions

### Phase 3: Documentation Overhaul

#### 3.1 New README Structure

```markdown
# CJE - Causal Judge Evaluation

Get unbiased policy estimates from biased judge scores.

## Install
pip install causal-judge-evaluation

## Quick Start
from cje import analyze_dataset
results = analyze_dataset("your_data.jsonl")
print(f"Best policy: {results.best_policy}")

## Learn More
- [Data Format](docs/data_format.md) - Required JSONL structure
- [Understanding Results](docs/results.md) - Interpreting outputs
- [Advanced Usage](docs/advanced.md) - Manual control
- [Research](docs/research.md) - Experimental features
```

#### 3.2 Progressive Documentation

1. **Quick Start** (1 page)
   - Installation
   - Basic usage
   - Data format

2. **User Guide** (5 pages)
   - Common workflows
   - Understanding diagnostics
   - Troubleshooting

3. **Advanced** (10 pages)
   - Manual workflows
   - Custom configuration
   - Performance tuning

4. **Research** (unlimited)
   - Theory
   - Experimental features
   - Contributing

### Phase 4: Smart Defaults Implementation

#### 4.1 Auto Estimator Selection

```python
def _choose_estimator(dataset, fresh_draws_available):
    """Smart estimator selection based on data characteristics."""
    
    # Quick data scan
    ess_estimate = _estimate_ess(dataset)
    oracle_coverage = _get_oracle_coverage(dataset)
    
    if fresh_draws_available and ess_estimate < 0.1:
        return "dr"  # Poor overlap, use DR
    elif oracle_coverage < 0.05:
        logger.warning("Low oracle coverage, results may be unreliable")
        return "ips"
    else:
        return "ips"  # Good overlap, IPS is sufficient
```

#### 4.2 Auto Robustness Selection

```python
def _choose_robustness(dataset, initial_diagnostics):
    """Choose robustness level based on data characteristics."""
    
    if initial_diagnostics.tail_index < 2:
        return "high"  # Heavy tails detected
    elif initial_diagnostics.ess < 0.2:
        return "high"  # Poor overlap
    elif initial_diagnostics.ess > 0.5:
        return "low"   # Good overlap
    else:
        return "medium"
```

## Migration Strategy

### Phase 1: Preparation (Week 1)
- [ ] Create comprehensive test suite for new API
- [ ] Identify what to keep vs remove
- [ ] Design final API surface

### Phase 2: Implementation (Week 2)
- [ ] Implement three-layer architecture
- [ ] Add smart defaults
- [ ] Create presets system
- [ ] Move examples to examples/

### Phase 3: Documentation (Week 3)
- [ ] Rewrite README for simplicity
- [ ] Create progressive documentation
- [ ] Add decision trees for estimator choice
- [ ] Create troubleshooting guide

### Phase 4: Testing & Polish (Week 4)
- [ ] Test all workflows
- [ ] Performance benchmarks
- [ ] User testing with colleagues
- [ ] Final polish

## Success Metrics

1. **API Simplicity**
   - Top-level exports: 50+ → 5
   - Estimator choices: 7 → 2 (+auto)
   - Configuration parameters: 15+ → 2

2. **Documentation**
   - Quick start: 1 page (currently 5+)
   - Time to first result: <5 minutes
   - Clear progression from simple to advanced

3. **User Experience**
   - Zero configuration for 80% use case
   - Clear error messages
   - Obvious next steps when things fail

## Risk Mitigation

Since we have no existing users, we can make a clean break:

1. **No Legacy Support**
   - Clean, simple API from day one
   - No confusion from multiple ways to do things
   - No technical debt from backwards compatibility

2. **Extensive Testing**
   - Test all new APIs thoroughly
   - Ensure correct results
   - Performance benchmarks

3. **Clear Versioning**
   - Version 0.2.0: New simplified API
   - Clear documentation of changes
   - No deprecation period needed

## Decision Points

### Critical Decisions Needed

1. **Estimator Naming**
   - Option A: `ips`, `dr` (simple, clear)
   - Option B: `calibrated-ips`, `doubly-robust` (descriptive)
   - **Recommendation**: Option A for simplicity

2. **Default Behavior**
   - Option A: `estimator="auto"` by default
   - Option B: `estimator="ips"` by default
   - **Recommendation**: Option A for best results

3. **Research Features**
   - Option A: Hide in `cje.research`
   - Option B: Keep in main but document as advanced
   - **Recommendation**: Option A for clarity

4. **Breaking Changes**
   - Option A: Clean break, new API only ✓
   - Option B: ~~Deprecation period with both APIs~~
   - **Decision**: Option A - no users, no legacy

## Next Steps

1. **Review & Approve Plan**
   - Get stakeholder feedback
   - Adjust based on concerns

2. **Create Feature Branch**
   - `git checkout -b simplification`
   - Implement changes incrementally

3. **Test Thoroughly**
   - Unit tests for all changes
   - Integration tests for workflows
   - User acceptance testing

4. **Document Everything**
   - Migration guide
   - New tutorials
   - API reference

## Conclusion

This simplification plan will transform CJE from a complex research tool into a simple, powerful library that's easy to start with but can grow with user needs. The key is **progressive disclosure** - show only what users need when they need it.

The lack of existing users gives us a unique opportunity to get this right before launch. We should take advantage of this freedom while being thoughtful about future compatibility.
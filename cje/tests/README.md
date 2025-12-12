# CJE Test Suite

## Overview

The CJE test suite focuses on end-to-end testing with real data. The suite consists of 14 test files (144 tests) providing comprehensive coverage of critical functionality.

## File Structure

```
tests/
├── conftest.py                           # Shared fixtures and arena data loaders
├── run_all_tests.py                     # Test runner script
│
├── E2E Tests (User Workflows)
│   ├── test_e2e_estimators.py           # Complete pipelines for all estimators
│   ├── test_e2e_features.py             # SIMCal, cross-fitting, OUA
│   ├── test_interface_integration.py    # High-level API testing
│   └── test_examples.py                 # Tutorial notebook and quickstart validation
│
├── Core Tests (Infrastructure)
│   ├── test_infrastructure.py           # Critical infrastructure and edge cases
│   ├── test_unified_folds.py            # Comprehensive fold management
│   ├── test_mc_variance.py              # Monte Carlo variance testing
│   └── test_reproducibility.py          # Determinism and seed propagation
│
├── Feature Tests
│   ├── test_covariates.py               # Calibration covariates
│   ├── test_data_loaders.py             # Data loading functions
│   ├── test_calibration_data_smoke.py   # calibration_data_path parameter
│   ├── test_oua_at_full_coverage.py     # OUA skipping at 100% coverage
│   ├── test_transport_diagnostics.py    # Transportability probe protocol
│   └── test_cle_diagnostics.py          # CLE and TTC diagnostics
│
└── data/                                 # Test datasets
    ├── arena_sample/                     # Real Arena 5K subset (1000 samples)
    │   ├── logged_data.jsonl             # Main dataset with judge scores
    │   └── fresh_draws/                  # Fresh draws for DR estimation
    └── *.jsonl                           # Synthetic test data for edge cases
```

## Core Concepts

### 1. End-to-End Focus
Instead of testing individual functions, we test complete pipelines:
- Load data → Calibrate → Create sampler → Estimate → Validate results
- All E2E tests use real Arena data for authentic testing
- Tests verify user-visible outcomes, not implementation details

### 2. Arena Sample Data
Real subset from Arena 5K evaluation:
- 1000 samples with actual judge scores and oracle labels
- 4 target policies: base, clone, parallel_universe_prompt, unhelpful
- Fresh draws for each policy enabling DR estimation
- Ground truth for validation (48% oracle coverage in base policy for AutoCal-R)

**Note**: The same arena sample data is used in `examples/arena_sample/` for the tutorial notebook and quickstart script.

### 3. Fixture Architecture
Shared fixtures in `conftest.py` provide consistent test data:
- **arena_sample**: Real 100-sample Arena dataset
- **arena_fresh_draws**: Filtered fresh draws matching dataset prompts
- **arena_calibrated**: Pre-calibrated Arena dataset
- **synthetic datasets**: Edge case testing (NaN, extreme weights)

### 4. Test Philosophy
- **Real Data Priority**: Use arena sample for integration tests
- **Complete Workflows**: Test what users actually do
- **Fast Feedback**: Most tests run in < 1 second
- **Clear Intent**: Each test has one clear purpose
- **Example Validation**: `test_examples.py` ensures tutorial notebook and quickstart work correctly

## Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Run E2E tests only (recommended for quick validation)
poetry run pytest cje/tests/test_e2e*.py -q

# Run specific test files
poetry run pytest cje/tests/test_e2e_estimators.py -v
poetry run pytest cje/tests/test_unified_folds.py
poetry run pytest cje/tests/test_examples.py  # Validate tutorial and examples

# Run with markers
poetry run pytest cje/tests -m e2e
poetry run pytest cje/tests -m "not slow"

# With coverage
poetry run pytest --cov=cje --cov-report=html cje/tests/

# Quick health check (single E2E test)
poetry run pytest cje/tests/test_e2e_estimators.py::TestE2EEstimators::test_calibrated_ips_pipeline -v
```

## Writing New Tests

When adding tests, follow these guidelines:

1. **Prefer E2E tests** - Test complete workflows
2. **Use arena data** - Real data finds real bugs
3. **Keep it focused** - Each test should have one clear purpose
4. **Document intent** - Clear test names and docstrings

```python
def test_new_feature_workflow(arena_sample):
    """Test that new feature improves estimates."""
    # 1. Calibrate dataset
    calibrated, cal_result = calibrate_dataset(
        arena_sample,
        judge_field="judge_score",
        oracle_field="oracle_label"
    )
    
    # 2. Create sampler
    sampler = PrecomputedSampler(calibrated)
    
    # 3. Run estimation with new feature
    estimator = YourEstimator(sampler, new_feature=True)
    results = estimator.fit_and_estimate()
    
    # 4. Validate results
    assert len(results.estimates) == 4  # 4 policies
    assert all(0 <= e <= 1 for e in results.estimates)
    # Test that new feature had expected effect
    assert results.metadata["new_feature_applied"] == True
```

## Key Design Decisions

### 1. **Real Data Testing**
Arena sample data provides ground truth validation:
- Catches regressions in production scenarios
- Tests all estimators with same data
- Reveals integration issues unit tests miss

### 2. **E2E Testing Priority**
Complete workflows over isolated functions:
- Test what users actually do
- Catch integration bugs
- Validate full pipelines
- Ensure components work together

### 3. **Unified Fold System**
Consistent cross-validation across all components:
- Hash-based fold assignment from prompt_id
- Prevents data leakage
- Ensures reproducibility
- Single source of truth (`data/folds.py`)

## Common Issues

### "FileNotFoundError for test data"
Ensure running from project root:
```bash
cd /path/to/causal-judge-evaluation
poetry run pytest cje/tests/
```

### "Slow test execution"
Skip slow tests during development:
```bash
poetry run pytest -m "not slow" cje/tests/
```

### "Import errors"
Install package in development mode:
```bash
poetry install
# or
pip install -e .
```

## Performance

- **E2E tests**: < 2 seconds each
- **Infrastructure tests**: < 1 second each
- **Full suite**: ~25 seconds for 111 tests

Test execution tips:
- Use `-x` to stop on first failure
- Use `-k pattern` to run tests matching pattern
- Use `--lf` to run last failed tests
- Use `-q` for quiet output during development
- Run E2E tests first for quick validation

## Summary

The CJE test suite contains 144 focused tests across 14 test files that validate real workflows with real data. This approach catches integration issues, runs fast, and provides comprehensive coverage of all estimators, calibration methods, diagnostic tools, covariates, data loading, and reproducibility guarantees. The `test_examples.py` file ensures the tutorial notebook and quickstart script remain accurate and functional.
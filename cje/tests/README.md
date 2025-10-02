# CJE Test Suite

## Overview

The CJE test suite has been radically simplified to focus on end-to-end testing with real data. We've reduced from 28 test files (238 tests) to 7 core test files (~80 tests) while maintaining comprehensive coverage of critical functionality.

## File Structure

```
tests/
├── conftest.py                    # Shared fixtures and arena data loaders
├── run_all_tests.py              # Test runner script
│
├── E2E Tests                    
│   ├── test_e2e_estimators.py    # Complete pipelines for all estimators
│   ├── test_e2e_features.py      # SIMCal, cross-fitting
│   └── test_interface_integration.py # High-level API testing
│
├── Core Tests
│   ├── test_infrastructure.py    # Critical infrastructure and edge cases
│   ├── test_unified_folds.py     # Comprehensive fold management
│   └── test_mc_variance.py       # Monte Carlo variance testing
│
└── data/                          # Test datasets
    ├── arena_sample/              # Real Arena 10K subset (100 samples)
    │   ├── logged_data.jsonl      # Main dataset with judge scores
    │   └── fresh_draws/           # Fresh draws for DR estimation
    └── *.jsonl                    # Synthetic test data for edge cases
```

## Core Concepts

### 1. End-to-End Focus
Instead of testing individual functions, we test complete pipelines:
- Load data → Calibrate → Create sampler → Estimate → Validate results
- All E2E tests use real Arena data for authentic testing
- Tests verify user-visible outcomes, not implementation details

### 2. Arena Sample Data
Real subset from Arena 10K evaluation:
- 100 samples with actual judge scores and oracle labels
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Fresh draws for each policy enabling DR estimation
- Ground truth for validation

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

## Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Run E2E tests only (recommended for quick validation)
poetry run pytest cje/tests/test_e2e*.py -q

# Run specific test files
poetry run pytest cje/tests/test_e2e_estimators.py -v
poetry run pytest cje/tests/test_unified_folds.py

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

### 1. **Simplified Test Suite**
Reduced from 238 tests to ~80 focused tests:
- 73% reduction in test count
- Comprehensive coverage maintained
- Faster execution and easier maintenance
- Focus on integration over unit testing

### 2. **Real Data Testing**
Arena sample data provides ground truth validation:
- Catches regressions in production scenarios
- Tests all estimators with same data
- Reveals integration issues unit tests miss

### 3. **E2E Testing Priority**
Complete workflows over isolated functions:
- Test what users actually do
- Catch integration bugs
- Validate full pipelines
- Ensure components work together

### 4. **Unified Fold System**
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
- **Full suite**: ~15 seconds for all tests

Test execution tips:
- Use `-x` to stop on first failure
- Use `-k pattern` to run tests matching pattern
- Use `--lf` to run last failed tests
- Use `-q` for quiet output during development
- Run E2E tests first for quick validation

## Summary

The CJE test suite has been transformed from 238 scattered unit tests to ~80 focused tests that test real workflows with real data. This simplified approach catches more integration issues, runs faster, and is easier to maintain while providing comprehensive coverage of all estimators, calibration methods, and diagnostic tools.
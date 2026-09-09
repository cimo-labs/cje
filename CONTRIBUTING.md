# Contributing to CJE

Thanks for contributing! We keep things simple.

## 🎯 Core Principles

**Do One Thing Well** - Each component should have a single, clear purpose.

**Philosophy**:
- Simple > Clever
- Explicit > Magic
- YAGNI (You Aren't Gonna Need It)
- Fail fast with clear errors
- No duplicate utilities - search before implementing

## 🛠️ Setup

```bash
poetry install
poetry run pytest  # Verify everything works
```

Notebook execution dependencies are part of the development environment; notebook tests
must execute rather than skip for missing packages. For the optional research examples:

```bash
poetry install --with research
make test-examples  # Real notebook kernels, documented calls, and experiment regressions
```

CI executes these examples on every pull request, including the planning notebook's
production configuration and the documented bootstrap workflow. Release qualification
also includes the research tests.

## 📝 Code Standards

1. **Type everything** - Use type hints
2. **No magic values** - Return None or raise exceptions
3. **Single responsibility** - Each function does ONE thing
4. **Test your code** - All PRs need tests

## 🧪 Testing Philosophy

CJE prioritizes **statistical correctness** over code coverage metrics:

- **Monte Carlo validation** - Tests should verify statistical properties (unbiasedness, coverage)
- **Real data testing** - Test with actual arena data when possible
- **Mathematical verification** - Validate key properties (monotonicity, mean preservation)
- **Integration over unit tests** - Complete pipelines matter more than isolated functions

We don't optimize for coverage percentages. A test that validates statistical properties is worth more than 100 tests of getters/setters.

## ✅ Pull Request Checklist

Before submitting:
```bash
poetry run pytest           # Tests pass
poetry run mypy cje         # Types check
poetry run black cje        # Code formatted
```

PR Title: Use `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

## ❌ What We Don't Accept

- Workflow orchestration (that's the user's job)
- Retry logic or state management
- Unnecessary abstractions
- Clever code that's hard to understand
- Magic values (-999, -100, etc.)

## 💡 What We Need

- Performance optimizations (with benchmarks)
- Better diagnostic visualizations  
- Notebook examples
- Bug fixes

## 🔒 GitHub Settings (For Maintainers)

### Branch Protection on `main`
- Require 1 PR review
- Status checks must pass: `pytest`, `mypy`, `black`
- No force pushes
- No direct commits

### Merge Strategy
- Squash merge only (clean history)
- No merge commits or rebasing

---

**Thank you for contributing! Keep it simple. 🎯**

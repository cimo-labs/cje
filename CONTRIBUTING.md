# Contributing to CJE

Thanks for contributing! We keep things simple.

## 🎯 Core Principle

**Do One Thing Well** - Read [CLAUDE.md](CLAUDE.md) for our philosophy.

**TL;DR**: Simple > Clever. Explicit > Magic. YAGNI.

## 🛠️ Setup

```bash
poetry install
poetry run pytest  # Verify everything works
```

## 📝 Code Standards

1. **Type everything** - Use type hints
2. **No magic values** - Return None or raise exceptions  
3. **Single responsibility** - Each function does ONE thing
4. **Test your code** - All PRs need tests

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
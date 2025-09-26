# Overlap Metrics Integration Plan

## Executive Summary

This document outlines an **ambitious yet disciplined** integration strategy for advanced overlap metrics into the CJE system. We will build a comprehensive overlap analysis system that fundamentally improves how CJE handles distribution mismatch.

**Core Innovation**: Decomposing overlap into unfixable (Hellinger) and fixable (ESS) components, with thresholds tied to desired confidence interval widths.

**Vision**: Make overlap analysis a first-class concept that orchestrates method selection, calibration configuration, and user communication throughout CJE.

**Timeline**: 4 phases over 3-4 months with validation gates between phases to ensure we're building the right thing.

## Background

### Problem Statement

Current CJE diagnostics use ad-hoc ESS thresholds (10%, 30%) without theoretical justification. These thresholds:
- Don't account for sample size
- Can't distinguish structural mismatch from variance inflation
- Can be artificially improved by calibration, hiding real problems
- Aren't tied to user requirements (desired precision)

### Solution Overview

We implement three complementary metrics:
1. **Hellinger Affinity**: Measures structural overlap (cannot be gamed)
2. **ESS with Auto-tuning**: Statistical efficiency tied to CI width goals
3. **Hill Tail Index**: Detects pathological weight distributions

The key insight: Hellinger tells us whether to give up, ESS tells us how hard to try.

## Current State (Completed)

### What We've Built
- ✅ Core implementation in `cje/diagnostics/overlap.py`
- ✅ Integration with `IPSDiagnostics` and `CalibratedIPS`
- ✅ Comprehensive test suite
- ✅ Documentation and demo scripts

### Architecture
```
Weights → CalibratedIPS → Diagnostics → Hellinger/ESS/Tail
                              ↓
                      Status determination
```

### Limitations
- Computed late in pipeline (during diagnostics)
- Global metrics only (no stratification)
- Descriptive only (doesn't affect decisions)
- No caching (recomputed each time)

## Proposed Architecture

### Design Principles
1. **Separation of Concerns**: Overlap analysis independent of estimation
2. **Compute Once**: Cache and reuse expensive computations
3. **Progressive Enhancement**: Add capabilities without breaking changes
4. **User Control**: Automated decisions can be overridden

### Component Hierarchy
```
┌─────────────────────────────────────┐
│         OverlapAnalyzer             │  First-class component
│   (Pre-flight overlap assessment)   │  Computes once, caches
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│   Method     │ │  Calibration │     Uses overlap to configure
│  Selector    │ │   Adapter    │     Adaptive based on metrics
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────────────────────┐
│        Estimators            │       Receive configuration
│  (IPS, CalibratedIPS, DR)    │       Include metrics in results
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        Diagnostics           │       Surface overlap prominently
│   (Include overlap metrics)  │       Provide interpretations
└──────────────────────────────┘
```

## Implementation Strategy

### Design Principles for Disciplined Execution

1. **Empirical Validation Gates**: Each phase must demonstrate measurable improvement
2. **Clean Architecture**: Maintain separation of concerns and single responsibility
3. **Performance Budgets**: Each new feature must stay within defined performance bounds
4. **User-Centric Design**: Features must map to clear user needs, not theoretical elegance
5. **Reversibility**: Design for easy rollback if features don't provide value
6. **Comprehensive Testing**: 100% coverage for core paths, property-based testing for edge cases
7. **Documentation-Driven**: Write docs first, implement second

### Phase 0: Foundation (Weeks 1-2) ✅ COMPLETED

**What We've Already Built**:
- Core implementation in `cje/diagnostics/overlap.py`
- Hellinger affinity computation with numerical stability
- Auto-tuned threshold calculation tied to CI widths
- Integration with `IPSDiagnostics` and `CalibratedIPS`
- Comprehensive test suite with 22 passing tests
- Demo script showing practical usage

**Validation Gate**: ✅ All tests pass, demo runs successfully

### Phase 1: Measurement Infrastructure (Weeks 3-4)

**Goal**: Establish `OverlapAnalyzer` as a first-class component with robust measurement

**Deliverables**:
1. **OverlapAnalyzer Component**
   ```python
   class OverlapAnalyzer:
       def __init__(self, sampler: BaseSampler):
           self.sampler = sampler
           self._cache = {}
           self._stratified_cache = {}
       
       def analyze_policy(self, policy: str) -> OverlapMetrics:
           """Compute comprehensive overlap metrics."""
       
       def analyze_all_policies(self) -> Dict[str, OverlapMetrics]:
           """Batch analysis with progress reporting."""
       
       def stratified_analysis(self, policy: str, 
                              stratify_by: str = "judge_score",
                              n_bins: int = 10) -> List[OverlapMetrics]:
           """Reveal local overlap problems."""
   ```

2. **Integration Points**
   - Add to `analyze_dataset()` as optional pre-flight check
   - Include in ablation study outputs
   - Log metrics for all estimations

3. **Monitoring**
   - Track correlation between overlap metrics and estimation errors
   - Identify edge cases where global metrics mislead
   - Gather data on typical overlap ranges

**Validation Gate** (must pass to proceed to Phase 2):
- ✓ Zero breaking changes to existing code
- ✓ Metrics computed for >1000 policy evaluations  
- ✓ Performance overhead <5% (measured via profiling)
- ✓ Correlation analysis shows Hellinger adds predictive value beyond ESS
- ✓ At least 3 real examples where overlap metrics would have prevented failures

### Phase 2: Conditional Analysis (Weeks 3-4)

**Goal**: Address the global/local problem with stratified metrics

**Deliverables**:
1. **Stratified Overlap Metrics**
   ```python
   def compute_conditional_overlap(
       weights: np.ndarray,
       conditions: np.ndarray,  # E.g., judge scores, rewards
       n_strata: int = 10
   ) -> List[OverlapMetrics]:
       """Compute overlap within strata."""
   ```

2. **Risk-Weighted Overlap**
   ```python
   def compute_risk_weighted_overlap(
       weights: np.ndarray,
       importance_scores: np.ndarray  # E.g., reward magnitude
   ) -> OverlapMetrics:
       """Weight overlap by decision importance."""
   ```

3. **Visualization Tools**
   - Heatmaps showing overlap vs judge score
   - Identification of problem regions
   - Cumulative overlap curves

**Validation Gate** (must pass to proceed to Phase 3):
- ✓ Identify at least 3 cases where global metrics mislead
- ✓ Stratified metrics reveal local problems invisible to global metrics
- ✓ Visualizations clearly communicate overlap structure to users
- ✓ Performance remains within 10% overhead budget
- ✓ User study shows improved understanding of failures

### Phase 3: Adaptive Configuration (Weeks 5-6)

**Goal**: Use overlap metrics to configure the system

**Deliverables**:
1. **Method Selection Logic**
   ```python
   class AdaptiveMethodSelector:
       def select_method(self, 
                        overlap: OverlapMetrics,
                        available_methods: List[str],
                        constraints: Dict) -> str:
           """Choose optimal estimation method."""
   ```

2. **SIMCal Auto-Configuration**
   ```python
   def adapt_simcal_config(overlap: OverlapMetrics) -> SimcalConfig:
       """Set calibration aggressiveness based on overlap."""
       
       if overlap.hellinger_affinity < 0.25:
           # Aggressive calibration for poor overlap
           return SimcalConfig(var_cap_rho=10.0, ess_floor=0.05)
       elif overlap.hellinger_affinity < 0.35:
           # Moderate calibration
           return SimcalConfig(var_cap_rho=5.0, ess_floor=0.10)
       else:
           # Conservative for good overlap
           return SimcalConfig(var_cap_rho=2.0, ess_floor=0.20)
   ```

3. **DR Component Weighting**
   ```python
   def compute_dr_weights(overlap: OverlapMetrics) -> Tuple[float, float]:
       """Balance IPW vs DM based on overlap quality."""
       if overlap.hellinger_affinity < 0.3:
           return 0.2, 0.8  # Rely mostly on outcome model
       else:
           return 0.5, 0.5  # Balanced
   ```

**Validation Gate** (must pass to proceed to Phase 4):
- ✓ Automated configuration improves estimates in >60% of poor-overlap cases
- ✓ No degradation in good-overlap cases (verified via A/B testing)
- ✓ User can override automated decisions with simple configuration
- ✓ Configuration logic is interpretable and documented
- ✓ Empirical validation of threshold choices (not just theoretical)

### Phase 4: System Integration (Weeks 7-8)

**Goal**: Make overlap analysis a core part of CJE workflow

**Deliverables**:
1. **Pre-flight Checks**
   ```python
   def preflight_overlap_check(sampler, policies) -> Dict[str, str]:
       """Determine viability before estimation."""
       decisions = {}
       for policy in policies:
           overlap = analyzer.analyze_policy(policy)
           if overlap.hellinger_affinity < 0.2:
               decisions[policy] = "refuse"
           elif overlap.hellinger_affinity < 0.35:
               decisions[policy] = "needs_dr"
           else:
               decisions[policy] = "proceed"
       return decisions
   ```

2. **Unified Reporting**
   - Overlap summary at top of all reports
   - Color-coded overlap quality indicators
   - Recommendations with explanations

3. **CLI Integration**
   ```bash
   # New overlap-first workflow
   cje analyze data.jsonl --overlap-check --auto-select-method
   
   # Overlap-only analysis
   cje overlap data.jsonl --stratify-by judge_score
   ```

**Success Criteria**:
- Overlap metrics appear in all estimation reports
- Pre-flight checks prevent >90% of catastrophic failures
- User satisfaction with automated decisions >80%

## Technical Considerations

### Performance
- **Caching Strategy**: Compute once per (sampler, policy) pair
- **Batch Computation**: Vectorize Hellinger computation across policies
- **Lazy Evaluation**: Only compute stratified metrics when requested
- **Memory Management**: Clear cache when sampler changes

### Backward Compatibility
- All new fields are optional in dataclasses
- Existing code paths unchanged
- New features behind feature flags initially
- Gradual deprecation of old thresholds

### Testing Strategy
1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Full pipeline with overlap analysis
3. **Regression Tests**: Ensure no degradation
4. **Stress Tests**: Performance with many policies
5. **Edge Cases**: Extreme weight distributions

## Maintaining Discipline

### Code Quality Standards
1. **Every PR must include**:
   - Tests with >95% coverage
   - Documentation updates
   - Performance benchmarks
   - Example usage

2. **Architecture Reviews**:
   - Design doc before implementation for major features
   - Code review by 2+ reviewers
   - Architecture decision records (ADRs) for key choices

3. **Continuous Validation**:
   - Weekly analysis of metric correlations
   - User feedback sessions every 2 weeks
   - Performance regression tests in CI

### Early Warning Signs (when to pivot)
- Hellinger doesn't correlate with estimation error after 1000+ examples
- Performance overhead exceeds 15% 
- Users find metrics confusing in >50% of feedback sessions
- Stratified analysis doesn't reveal new insights after Phase 2
- Auto-configuration degrades performance in A/B tests

### Success Amplifiers
- Create internal dashboard showing overlap metrics for all runs
- Regular "overlap clinic" sessions to review interesting cases
- Build corpus of examples where metrics prevented failures
- Publish internal blog posts explaining insights

## Risk Analysis

### Technical Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance degradation | Low | Medium | Caching, profiling |
| Breaking changes | Low | High | Extensive testing |
| Incorrect auto-config | Medium | Medium | Override options |
| Cache invalidation bugs | Medium | Low | Clear documentation |

### Adoption Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| User confusion | Medium | Medium | Clear documentation |
| Over-reliance on metrics | Medium | Medium | Education on limitations |
| Resistance to auto-config | Low | Low | Make it optional |

## Success Metrics

### Quantitative
- **Adoption**: >50% of analyses use overlap metrics within 3 months
- **Accuracy**: Reduce catastrophic failures by >80%
- **Performance**: Overhead <5% for typical workloads
- **Coverage**: Metrics computed for >95% of estimations

### Qualitative
- User feedback positive
- Clearer understanding of estimation failures
- Reduced debugging time
- Increased confidence in results

## Alternative Approaches Considered

### 1. Deep Integration (Rejected)
Embed overlap computation directly in estimators.
- **Pro**: Tighter integration
- **Con**: Violates separation of concerns, harder to test

### 2. External Service (Rejected)
Separate microservice for overlap analysis.
- **Pro**: Complete separation
- **Con**: Deployment complexity, latency

### 3. Lazy Computation (Rejected)
Compute overlap only when diagnostics requested.
- **Pro**: No upfront cost
- **Con**: Misses pre-flight check opportunity

## Open Questions

1. **Should overlap analysis be mandatory or optional?**
   - Start optional, make mandatory after validation

2. **How to handle time-varying policies?**
   - Future work: temporal overlap metrics

3. **Should we expose raw Hellinger or interpreted categories?**
   - Both: categories for users, values for developers

4. **How aggressive should auto-configuration be?**
   - Start conservative, increase based on empirical results

5. **Should stratified analysis be default?**
   - No, too expensive; make it opt-in

## Next Steps

### Immediate (Week 1)
1. Create `OverlapAnalyzer` class
2. Add caching infrastructure
3. Write comprehensive tests

### Short-term (Weeks 2-4)
1. Integrate with `analyze_dataset()`
2. Add stratified analysis
3. Begin collecting empirical data

### Medium-term (Weeks 5-8)
1. Implement adaptive configuration
2. Add pre-flight checks
3. Update documentation

### Long-term (3-6 months)
1. Make overlap analysis default
2. Deprecate fixed thresholds
3. Publish empirical validation

## Appendix: Mathematical Details

### Hellinger Affinity
For importance weights $w = \pi'/\pi_0$:
$$\mathcal{A} = \mathbb{E}_{x \sim \pi_0}[\sqrt{w(x)}] = \int \sqrt{p_{\pi'}(x) p_{\pi_0}(x)} dx$$

Properties:
- Range: $(0, 1]$ where 1 indicates perfect overlap
- Cannot be improved by weight calibration
- Related to Hellinger distance: $H = \sqrt{1 - \mathcal{A}^2}$

### Auto-tuned ESS Threshold
For desired CI half-width $\delta$ and sample size $n$:
$$\text{ESS}_{\text{threshold}} = \frac{0.9604}{n \cdot \delta^2}$$

Derivation from variance bound:
$$\text{Var}(\hat{V}_{\text{IPS}}) \leq \frac{1}{4n \cdot \text{ESS}_{\text{frac}}}$$
$$\text{CI half-width} \approx \frac{1.96}{2\sqrt{n \cdot \text{ESS}_{\text{frac}}}}$$

### Diagnostic Cascade
1. **Hellinger < 0.2**: Catastrophic (refuse)
2. **Hellinger ∈ [0.2, 0.35]**: Poor (calibration might help)
3. **ESS < auto-threshold**: Below precision target
4. **Tail index < 2**: Infinite variance risk
5. **Otherwise**: Proceed with confidence

## Revision History

- v1.0 (2024-01): Initial planning document
- v1.1 (2024-01): Added implementation of Phase 0 (foundation)
- v1.2 (2024-01): Revised to emphasize disciplined execution with validation gates
- v1.3 (TBD): Post-Phase-1 updates based on empirical findings
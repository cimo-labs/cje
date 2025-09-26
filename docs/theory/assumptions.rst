Key Assumptions
===============

This page details the key assumptions underlying CJE methods and what happens when they're violated.

Core Assumptions
----------------

**(D2) Overlap (Positivity)**
   :math:`\pi_0(a|x) > 0` whenever :math:`\pi'(a|x) > 0`
   
   - **Meaning**: The logging policy must have non-zero probability for any action the target policy might take
   - **Violation**: Infinite importance weights, undefined estimates
   - **Detection**: Check ESS in diagnostics
   - **Mitigation**: Use DR methods or collect more diverse data

**(J1) Oracle Slice**
   Oracle labels are a simple random subsample
   
   - **Meaning**: The subset with ground truth labels is representative
   - **Violation**: Biased calibration mapping
   - **Detection**: Compare oracle subset statistics to full data
   - **Mitigation**: Ensure random sampling for oracle labeling

**(J2-M) Judge Monotone Sufficiency**
   :math:`\mathbb{E}[Y|S]` is monotone, :math:`\mathbb{E}[W|S]` is monotone
   
   - **Meaning**: Higher judge scores correspond to higher oracle values and importance weights
   - **Violation**: SIMCal may increase variance
   - **Detection**: Check calibration plots
   - **Mitigation**: Use raw weights or different ordering index

**(R3) DR Convergence Rates**
   Either weights or outcome model converges at :math:`n^{-1/4}`
   
   - **Meaning**: At least one nuisance function is well-estimated
   - **Violation**: Slower than √n convergence
   - **Detection**: Cross-validation performance
   - **Mitigation**: Improve model specification or use simpler methods

Robustness to Violations
-------------------------

**CalibratedIPS**
   - Robust to: Outcome model misspecification (doesn't use one)
   - Sensitive to: Poor overlap, non-monotone judge scores
   
**DR Methods**
   - Robust to: Either weight OR outcome model misspecification
   - Sensitive to: Both being misspecified simultaneously
   
**TMLE**
   - Robust to: Model misspecification (targeted updates)
   - Sensitive to: Extreme weights, poor initial fits

Diagnostic Checks
-----------------

Always check these diagnostics:

1. **ESS** - Effective sample size (should be > 10% typically)
2. **Hill Index** - Tail behavior (should be > 2)
3. **Calibration R²** - Judge-oracle alignment (should be > 0.5)
4. **Orthogonality Score** - For DR methods (CI should contain 0)

See :doc:`/modules/diagnostics` for comprehensive diagnostic documentation.
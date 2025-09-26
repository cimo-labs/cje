SIMCal: Stacked Score-Indexed Monotone Calibration  
===================================================

Overview
--------

SIMCal (Stacked Score-Indexed Monotone Calibration) is the core weight stabilization technique in CJE. It combines multiple candidate weight vectors ({baseline, increasing, decreasing}) via convex optimization to minimize out-of-fold (OOF) influence function variance.

The Problem with Raw Importance Weights
----------------------------------------

Standard importance sampling uses weights:

.. math::

   w_i = \frac{\pi'(a_i|x_i)}{\pi_0(a_i|x_i)}

These weights can be extremely variable, leading to:

- **Infinite variance** when support mismatch exists
- **Effective sample size collapse** (ESS → 1)
- **Unstable estimates** that change wildly with small data perturbations

The SIMCal Solution
-------------------

SIMCal addresses weight instability through a two-stage process:

1. **Stacking**: Combines {baseline, increasing, decreasing} monotone projections to minimize OOF variance
2. **Blending**: Applies uniform shrinkage to meet ESS/variance constraints

Implementation Algorithm
------------------------

Based on the actual code in ``cje/calibration/simcal.py``:

1. Build Candidate Weight Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Three candidates are created:
   # 1. Baseline (optional): Raw weights
   # 2. Increasing: Isotonic regression with increasing=True
   # 3. Decreasing: Isotonic regression with increasing=False
   
   candidates = []
   if include_baseline:
       candidates.append(raw_weights)  # "baseline"
   
   iso_inc = IsotonicRegression(increasing=True)
   w_inc = iso_inc.fit(scores, raw_weights).predict(scores)
   candidates.append(w_inc)  # "increasing"
   
   iso_dec = IsotonicRegression(increasing=False)
   w_dec = iso_dec.fit(scores, raw_weights).predict(scores)
   candidates.append(w_dec)  # "decreasing"

2. Compute Out-of-Fold Influence Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation computes OOF influence functions for each candidate:

.. code-block:: python

   # For each fold and candidate, compute:
   # IF = w * residual - mean(w * residual on training folds)
   
   # The residual depends on what's available:
   # - DR mode: residuals = rewards - g_oof(scores)  
   # - IPS mode: residuals = rewards
   # - Weight-only mode: residuals = 1 (just calibrate weights)

3. Solve Quadratic Program on Simplex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finds optimal mixture weights π that minimize variance:

.. code-block:: python

   # min_π π^T Σ π  subject to  π ≥ 0, Σπ = 1
   # where Σ is the empirical covariance of influence functions
   
   mixture_weights = solve_simplex_qp(covariance_matrix)

4. Apply Constraints via Uniform Blending
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After stacking, applies a single γ-blend toward uniform to meet constraints:

.. code-block:: python

   # Blend toward uniform to satisfy ESS floor or variance cap
   w_final = (1 - γ) * w_stacked + γ * 1
   
   # γ chosen to exactly satisfy the binding constraint

5. Optional Baseline Shrinkage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final stability enhancement:

.. code-block:: python

   # Shrink slightly toward baseline for numerical stability
   w_final = (1 - baseline_shrink) * w_final + baseline_shrink * 1

Configuration Options
---------------------

The ``SimcalConfig`` class controls calibration behavior:

.. code-block:: python

   from cje.calibration.simcal import SimcalConfig
   
   config = SimcalConfig(
       ess_floor=0.2,           # Min ESS as fraction of n (default 0.2)
       var_cap=None,            # Max variance (default None = no cap)
       include_baseline=True,   # Include raw weights in stack
       baseline_shrink=0.05,    # Final shrinkage toward uniform
       n_folds=5,               # Folds for OOF computation
       ridge_lambda=1e-8        # Ridge regularization for stability
   )

Key parameters:

- ``ess_floor``: Ensures ESS ≥ ess_floor * n (e.g., 0.2 means 20% minimum ESS)
- ``var_cap``: Hard cap on weight variance (e.g., 1.0 means no variance increase)
- ``include_baseline``: Whether to include raw weights as a candidate
- ``baseline_shrink``: Final shrinkage toward uniform (0.05 = 5% uniform blend)

Usage in CalibratedIPS
----------------------

The ``CalibratedIPS`` estimator automatically applies SIMCal:

.. code-block:: python

   from cje import CalibratedIPS, PrecomputedSampler
   
   sampler = PrecomputedSampler(dataset)
   estimator = CalibratedIPS(
       sampler,
       ess_floor=0.2,        # 20% minimum ESS
       var_cap=None,         # No explicit variance cap
       include_baseline=True,# Include raw weights
       baseline_shrink=0.05  # 5% uniform shrinkage
   )
   results = estimator.fit_and_estimate()

The estimator:
1. Computes raw importance weights for each policy
2. Applies SIMCal calibration using judge scores as the ordering index
3. When a calibrator is available, uses cross-fitted predictions g_oof(S) as the ordering index
4. Stores calibration info in ``_calibration_info`` for diagnostics

DR-Aware Calibration
--------------------

When used with DR estimators, SIMCal becomes DR-aware:

.. code-block:: python

   # If calibrator with cross-fitted models is available:
   if calibrator and hasattr(calibrator, 'predict_oof'):
       g_oof = calibrator.predict_oof(judge_scores, fold_ids)
       residuals = rewards - g_oof  # DR residuals
       ordering_index = g_oof        # Use g_oof as ordering
   else:
       residuals = rewards           # IPS residuals  
       ordering_index = judge_scores # Use judge scores

This aligns the monotone projection with the actual nuisance function used in DR.

Diagnostics and Information
---------------------------

SIMCal returns detailed calibration information:

.. code-block:: python

   # Access calibration info from estimator
   calib_info = estimator._calibration_info[policy]
   
   print(f"Mixture weights: {calib_info['mixture_weights']}")
   # e.g., {'baseline': 0.1, 'increasing': 0.7, 'decreasing': 0.2}
   
   print(f"Gamma (uniform blend): {calib_info['gamma']:.3f}")
   print(f"Variance before: {calib_info['var_before']:.3f}")
   print(f"Variance after: {calib_info['var_after']:.3f}")
   print(f"ESS before: {calib_info['ess_before']:.1%}")
   print(f"ESS after: {calib_info['ess_after']:.1%}")

Interpreting mixture weights:
- High ``increasing`` weight: Positive correlation between scores and importance
- High ``decreasing`` weight: Negative correlation
- High ``baseline`` weight: Monotone projections not helpful

ESS Floor vs Variance Cap
-------------------------

The relationship between ESS and variance:

.. math::

   ESS = \frac{n}{1 + \text{Var}(w)}

This implies:
- ``ess_floor=0.2`` is equivalent to ``var_cap=4.0``
- ``ess_floor=0.5`` is equivalent to ``var_cap=1.0``
- ``ess_floor=1.0`` is equivalent to ``var_cap=0.0`` (uniform weights)

If both are specified, the tighter constraint applies.

Computational Complexity
------------------------

- **Isotonic regression**: O(n log n) per candidate
- **OOF computation**: O(n * K * F) where K=candidates, F=folds
- **Quadratic program**: O(K³) where K ≤ 3
- **Overall**: O(n log n) dominated by isotonic regression

Practical Guidelines
--------------------

Default Settings (Conservative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Good for most use cases
   CalibratedIPS(sampler, ess_floor=0.2)

High Variance Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Tighter constraints for stability
   CalibratedIPS(sampler, ess_floor=0.3, baseline_shrink=0.1)

Large Sample Size
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Can afford less constraint
   CalibratedIPS(sampler, ess_floor=0.1, baseline_shrink=0.01)

Debugging Weight Issues
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check what SIMCal is doing
   estimator = CalibratedIPS(sampler)
   estimator.fit()
   
   for policy in sampler.target_policies:
       info = estimator._calibration_info[policy]
       print(f"\n{policy}:")
       print(f"  Stacking: {info['mixture_weights']}")
       print(f"  Uniform blend: {info['gamma']:.1%}")
       print(f"  Variance reduction: {1 - info['var_after']/info['var_before']:.1%}")

Limitations
-----------

1. **Assumes monotone relationship**: Between scores and importance weights
2. **Requires meaningful scores**: Random scores won't help
3. **Computational overhead**: ~20-30% slower than raw IPS
4. **Not unbiased**: Trades bias for variance reduction

When SIMCal May Not Help
------------------------

- **Perfect overlap**: All weights ≈ 1
- **Random judge scores**: No signal for monotone projection  
- **Very large n**: Variance less of a concern
- **Need strict unbiasedness**: Use RawIPS instead

References
----------

- The stacked calibration approach is based on convex optimization for ensemble learning
- Isotonic regression uses scikit-learn's implementation (PAVA algorithm)
- See ``cje/calibration/simcal.py`` for complete implementation details
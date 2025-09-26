Causal Judge Evaluation (CJE) with SIMCal
==========================================

.. image:: img/CJE_logo.svg
   :align: center
   :alt: CJE Logo
   :width: 400px

**Shape-Constrained, Unbiased Off-Policy Metrics for LLM Systems and Beyond**

CJE transforms routine LLM evaluation logs into unbiased, variance-controlled estimates of counterfactual performance: *"What would our KPI be if we shipped policy π′ instead of π₀?"*

The Problem
-----------

Modern LLM evaluation relies on automatic judges (GPT-4, Claude, etc.) to score outputs at scale. But these offline metrics are **correlational**—computed under your logging policy π₀, they don't answer the **causal** question of how a new policy π′ would perform if deployed.

The Solution: SIMCal
--------------------

CJE recasts judge-based evaluation as **calibrated causal inference** using our novel **Surrogate-Indexed Monotone Calibration (SIMCal)**:

1. **Isotonic Reward Calibration**: Maps judge scores S to calibrated rewards R = f(S) using a small oracle slice
2. **Variance-Safe Weight Calibration**: Projects importance weights onto monotone functions indexed by the judge, with an explicit variance cap  
3. **Out-of-Fold Stacking**: Combines {baseline, increasing, decreasing} candidates to minimize influence-function variance
4. **Doubly Robust Estimation**: Achieves √n-rate inference when either nuisance converges at n^(-1/4)

Quick Start
-----------

.. code-block:: bash

   # Install from repository
   cd causal-judge-evaluation
   pip install -e .

.. code-block:: python

   from cje import analyze_dataset
   
   # One-line causal evaluation (with fresh draws for DR)
   results = analyze_dataset(
       "logs.jsonl",
       fresh_draws_dir="responses/"  # Required for stacked-dr default
       # Automatically uses all available oracle labels
   )
   
   # Or use calibrated-ips if no fresh draws available
   # results = analyze_dataset("logs.jsonl", estimator="calibrated-ips")
   
   # Get policy value estimate with 95% CI
   print(f"Policy value: {results.estimates[0]:.3f} ± {1.96 * results.standard_errors[0]:.3f}")

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   examples/index
   tutorials/dr_quickstart

.. toctree::
   :maxdepth: 2
   :caption: How‑to Guides

   howto/choose_estimator
   howto/fresh_draws
   howto/diagnostics
   howto/validate_and_troubleshoot
   howto/export_and_compare

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/how_it_works

.. toctree::
   :maxdepth: 2
   :caption: Reference

   modules/index
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index

Key Results
-----------

- **Variance Reduction**: SIMCal increases ESS (effective sample size) by construction through majorization
- **Mean Preservation**: All calibrations preserve the population mean exactly
- **Efficiency**: DR-CPO achieves semiparametric efficiency under standard conditions
- **Auditability**: Comprehensive diagnostics expose assumptions with quantitative gates

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

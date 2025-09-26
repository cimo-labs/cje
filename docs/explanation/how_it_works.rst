How It Works
============

This page gives a high-level view of the CJE pipeline and keeps the schematic diagram for reference.

Overview
--------

At a high level, CJE transforms biased judge scores into unbiased policy estimates via:

- Calibration: map judge scores S to rewards R = f(S) using a small oracle slice
- Weight calibration (SIMCal): stabilize importance weights with monotone projections and ESS/variance constraints
- Estimation: IPS or DR estimators with cross-fitting and diagnostics

Pipeline Schematic
------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │                            INPUT DATA                               │
   │  ┌─────────────────────────────────────────────────────────────┐    │
   │  │                    Logged Conversations                     │    │
   │  │  • Prompts (X)                                              │    │
   │  │  • Responses (A) from policy π₀                             │    │
   │  │  • Judge scores (S) from automatic evaluator                │    │
   │  │  • Log probabilities: log p_π₀(A|X)                         │    │
   │  └─────────────────────────────────────────────────────────────┘    │
   │                                │                                    │
   │          ┌─────────────────────┴──────────────────────┐             │
   │          ▼                                            ▼             │
   │  ┌───────────────┐                          ┌────────────────────┐  │
   │  │ Oracle Subset │                          │ Full Dataset       │  │
   │  │ (~5-10% data) │                          │ (100% of data)     │  │
   │  │ + Human labels│                          │ Judge scores only  │  │
   │  │     (Y)       │                          │                    │  │
   │  └───────────────┘                          └────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────┘
              │                                            │
              ▼                                            │
     ┌──────────────────────────┐                          │
     │   CALIBRATION STEP       │                          │
     │                          │                          │
     │ Learn f: S → Y via       │                          │
     │ isotonic regression      │                          │
     │ • Cross-fit with k folds │                          │
     │ • f^(-k) for fold k      │                          │
     └──────────────────────────┘                          │
              │                                            │
              └──────────────┬─────────────────────────────┘

   (Downstream: SIMCal weight calibration, IPS/DR estimation, diagnostics, and CIs)

Legend
------

- X = prompts, A = responses, S = judge scores, Y = oracle labels, R = rewards
- W = importance weights, W_c = calibrated weights, ψ̂ = influence function
- L = oracle label indicator, p = oracle coverage, m̂(S) = E[W|S], AUG = augmentation

Pointers
--------

- SIMCal details: :doc:`../theory/simcal`
- Estimators: :doc:`../modules/estimators`
- Diagnostics: :doc:`../modules/diagnostics`
- Quickstart: :doc:`../quickstart`

DR Quickstart (10 minutes)
=========================

This tutorial shows how to run a doubly‑robust (DR) analysis with fresh draws using the default calibrated stacked DR estimator (`stacked-dr`).

Prerequisites
-------------

- A logging dataset in JSONL with judge scores (see minimal schema in README)
- A directory of fresh draws (responses from target policies evaluated by the judge)

Command‑line (recommended)
--------------------------

.. code-block:: bash

   # Auto selects calibrated stacked DR when fresh draws are provided
   python -m cje analyze logs.jsonl \
     --fresh-draws-dir responses/ \
     --estimator auto \
     -o results.json

Python API
----------

.. code-block:: python

   from cje.interface.analysis import analyze_dataset

   results = analyze_dataset(
       dataset_path="logs.jsonl",
       estimator="stacked-dr",   # Default DR estimator (calibrated stacked DR)
       fresh_draws_dir="responses/",
   )

   # Per-policy results
   policies = results.metadata.get("target_policies", [])
   for i, p in enumerate(policies):
       est, se = results.estimates[i], results.standard_errors[i]
       print(f"{p}: {est:.3f} ± {se:.3f}")

Fresh Draws Layout
------------------

The helper loader expects per-policy JSONL or partitioned files. See :doc:`../howto/fresh_draws` for formats and tips.

Interpreting Results
--------------------

- DR diagnostics include outcome R² and influence‑function tail ratios
- Lower standard errors than IPS are common when fresh draws are informative
- If diagnostics show heavy tails or low ESS, consider collecting more data or revisiting policy overlap

Next Steps
----------

- Learn more about choosing estimators: :doc:`../howto/choose_estimator`
- Understand the pipeline: :doc:`../explanation/how_it_works`

Diagnostics & Gates
===================

What to Look At
---------------

- Weight ESS (effective sample size fraction)
- Hill tail index (heavy tails)
- Hellinger affinity (overlap)
- Outcome model R² (DR)

Quick Usage
-----------

.. code-block:: python

   d = results.diagnostics
   print(d.summary())
   
   if d and d.weight_ess < 0.1:
       print("Low ESS – consider tighter constraints or more data")
   if getattr(d, "tail_indices", None):
       worst = min(v for v in d.tail_indices.values() if v is not None)
       if worst < 2:
           print("Heavy tails – results may be unstable")

Reference Thresholds
--------------------

- ESS < 0.1 (10%) – very low; 0.1–0.3 – marginal; > 0.3 – safer
- Hill tail index α < 1.5 – extreme (mean risk); α < 2 – heavy (variance risk)
- Hellinger affinity – closer to 1 is better overlap

Mitigation Playbook
-------------------

- Increase overlap: choose policies with more similar behavior or add diversity to logging policy
- Use DR with fresh draws (``stacked-dr``) when available
- Tighten SIMCal constraints (higher ESS floor) or clip raw weights for baselines
- Collect more data; ensure stable folds via ``prompt_id``

See also
--------

- How it works: :doc:`../explanation/how_it_works`
- Estimator selection: :doc:`choose_estimator`

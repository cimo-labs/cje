Export & Compare
================

Export Results
--------------

CLI (JSON):

.. code-block:: bash

   python -m cje analyze logs.jsonl -o results.json

Python API:

.. code-block:: python

   from cje.utils.export import export_results_json
   export_results_json(results, "results.json")

Compare Policies
----------------

.. code-block:: python

   # Pairwise comparison with influence functions when available
   cmp = results.compare_policies(0, 1)
   print(cmp["difference"], cmp["p_value"], cmp["used_influence"]) 

Confidence Intervals
--------------------

.. code-block:: python

   lo, hi = results.confidence_interval(alpha=0.05)
   print(list(zip(results.metadata.get("target_policies", []), lo, hi)))

See also
--------

- API reference: :doc:`../api/index`

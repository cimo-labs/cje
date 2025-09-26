Examples
========

Working examples of CJE in action.

Arena 10K Analysis
------------------

See the ``experiments/arena_10k_simplified/`` directory for a complete end-to-end analysis pipeline using real Arena data.

Quick Examples
--------------

**Basic Analysis**

.. code-block:: python

   from cje import analyze_dataset
   
   results = analyze_dataset("data.jsonl", estimator="calibrated-ips")

**With Visualization**

.. code-block:: python

   from cje.visualization import plot_weight_dashboard_summary
   
   fig, metrics = plot_weight_dashboard_summary(
       raw_weights, 
       calibrated_weights
   )

For more examples, see the test suite in ``cje/tests/``.
Quick Start Guide
=================

This guide will get you up and running with CJE in minutes.

Installation
------------

Install CJE using pip:

.. code-block:: bash

   git clone https://github.com/fondutech/causal-judge-evaluation.git
   cd causal-judge-evaluation
   pip install -e .

Or with Poetry:

.. code-block:: bash

   poetry install

Basic Usage
-----------

The simplest way to use CJE is through the ``analyze_dataset`` function:

.. code-block:: python

   from cje import analyze_dataset
   
   # Analyze your logged data (with fresh draws for DR)
   results = analyze_dataset(
       "your_data.jsonl",
       fresh_draws_dir="responses/"  # Required for stacked-dr default
   )
   # Defaults to stacked-dr (most robust ensemble)
   
   # Or use calibrated-ips if no fresh draws available
   # results = analyze_dataset("your_data.jsonl", estimator="calibrated-ips")
   
   # View results
   print(f"Policy value: {results.estimates[0]:.3f}")
   print(f"Standard error: {results.standard_errors[0]:.3f}")
   print(f"Diagnostics: {results.diagnostics.summary()}")

Data Format
-----------

Your data should be in JSONL format with these required fields:

.. code-block:: json

   {
     "prompt": "What is machine learning?",
     "response": "Machine learning is...",
     "base_policy_logprob": -45.67,
     "target_policy_logprobs": {
       "gpt4": -42.31,
       "claude": -44.89
     },
     "metadata": {
       "judge_score": 0.82,
       "oracle_label": 0.90
     }
   }

See :doc:`modules/data` for detailed data format documentation.

Next Steps
----------

- :doc:`modules/index` - Module documentation
- :doc:`examples/index` - Examples
- :doc:`../howto/choose_estimator` - Choose the right estimator
- :doc:`theory/index` - Theory and background

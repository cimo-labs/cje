Validate & Troubleshoot
=======================

Validate a Dataset
------------------

CLI:

.. code-block:: bash

   python -m cje validate logs.jsonl -v

Checks:

- JSONL parsing and required fields
- Availability of base and target log probabilities
- Presence of judge scores and oracle labels (if any)

Common Issues
-------------

- Missing log probabilities → samples are filtered; high filter rates reduce power
- Very small n (<5 valid samples) → cross‑fitting may fail; gather more data
- Partial oracle labels → automatically handled via calibration; CIs include oracle uncertainty

Debug Tips
----------

- Print sampler summary and per‑policy valid counts
- Inspect diagnostics for ESS/tails; revisit policy overlap and constraints

See also
--------

- Diagnostics: :doc:`diagnostics`
- Fresh draws: :doc:`fresh_draws`

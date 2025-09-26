Choosing an Estimator
=====================

Default Recommendation
----------------------

**Use ``stacked-dr`` (the default)** — A robust ensemble of DR estimators (DR-CPO + TMLE + MRDR). This is the most reliable choice for production use when you have fresh draws available.

When to Use Other Estimators
-----------------------------

- **``calibrated-ips``** — When you need speed over robustness, or when computational resources are limited
- **``auto``** — Let CJE decide based on your setup (uses ``stacked-dr`` if fresh draws available, ``calibrated-ips`` otherwise)

Why DR is More Robust
---------------------

- **Double robustness**: Consistent if either weights OR outcome model is correct
- **Ensemble averaging**: ``stacked-dr`` combines multiple DR estimators, reducing variance
- **Better with poor overlap**: DR methods handle low ESS situations better than IPS
- **Reduced variance**: Outcome models reduce Monte Carlo variance in estimation

Advanced Notes
--------------

- ``raw-ips`` is available for baselines and diagnostics
- Individual DR estimators (``dr-cpo``, ``tmle``, ``mrdr``) are available for ablations
- SIMCal and calibration support are used where applicable in both IPS and DR paths

See also
--------

- DR quickstart: :doc:`../tutorials/dr_quickstart`
- How it works: :doc:`../explanation/how_it_works`

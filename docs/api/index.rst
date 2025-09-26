API Reference
=============

Main Entry Point
----------------

.. autofunction:: cje.analyze_dataset

Core Classes
------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   cje.data.models.Sample
   cje.data.models.Dataset
   cje.data.models.EstimationResult
   cje.data.precomputed_sampler.PrecomputedSampler

Estimators
----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   cje.estimators.calibrated_ips.CalibratedIPS
   cje.estimators.dr_base.DRCPOEstimator
   cje.estimators.mrdr.MRDREstimator
   cje.estimators.tmle.TMLEEstimator
   cje.estimators.stacking.StackedDREstimator

Calibration
-----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   cje.calibration.dataset.calibrate_dataset

For module‑level overviews, see :doc:`/modules/index`.

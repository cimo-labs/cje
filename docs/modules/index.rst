Module Documentation
====================

This section contains comprehensive documentation for each CJE module, organized by functionality.
Each module's documentation is maintained in its README file and included here for easy navigation.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules
   
   data
   calibration
   estimators
   diagnostics
   visualization
   
.. toctree::
   :maxdepth: 2
   :caption: Supporting Modules
   
   teacher_forcing
   utils

Overview
--------

The CJE framework is organized into focused modules, each with a specific responsibility:

- :doc:`data` - Data models, loading, and validation
- :doc:`calibration` - Judge calibration and weight calibration (SIMCal)
- :doc:`estimators` - All causal inference estimation methods
- :doc:`diagnostics` - Comprehensive diagnostic system
- :doc:`visualization` - Diagnostic plots and dashboards
- :doc:`teacher_forcing` - Log probability computation
- :doc:`utils` - Export and analysis utilities

Each module follows consistent patterns and interfaces, making it easy to compose them into complete analysis pipelines.
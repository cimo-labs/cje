Theory & Methods
================

Theoretical foundations and mathematical details of CJE.

.. toctree::
   :maxdepth: 2
   
   causal_inference
   simcal
   assumptions

Overview
--------

CJE is built on rigorous causal inference theory, combining:

- **Importance Sampling**: Reweighting logged data for counterfactual estimation
- **Calibration**: Mapping judge scores to oracle labels
- **SIMCal**: Variance reduction through monotone calibration
- **Doubly Robust Methods**: Combining outcome models with importance weights

Key Concepts
------------

**Causal Estimand**

We want to estimate:

.. math::

   V(\pi') = \mathbb{E}_{a \sim \pi'(\cdot|x)}[R(x, a)]

using data collected under logging policy :math:`\pi_0`.

**Importance Weights**

The importance weight for sample :math:`i` is:

.. math::

   W_i = \frac{\pi'(a_i|x_i)}{\pi_0(a_i|x_i)}

**SIMCal Projection**

SIMCal projects weights onto monotone functions of a surrogate index :math:`S`:

.. math::

   \tilde{W} = \arg\min_{W \in \mathcal{M}(S)} \text{Var}(W) \quad \text{s.t.} \quad \mathbb{E}[W] = 1

where :math:`\mathcal{M}(S)` is the set of monotone functions of :math:`S`.

See the following sections for detailed theory:

- :doc:`causal_inference` - Core causal inference framework
- :doc:`simcal` - SIMCal algorithm and theory
- :doc:`assumptions` - Key assumptions and their violations
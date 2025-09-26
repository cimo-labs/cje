Theory and Assumptions
=======================

This page provides the mathematical foundations of CJE, including key assumptions, theoretical results, and proofs of the main theorems.

Problem Setup
-------------

We observe data from a logging policy π₀:

.. math::

   \mathcal{D} = \{(x_i, a_i, r_i, s_i)\}_{i=1}^n

Where:
- :math:`x_i` : context/prompt
- :math:`a_i \sim \pi_0(\cdot|x_i)` : action/response from logging policy
- :math:`r_i` : reward (possibly calibrated from judge score)
- :math:`s_i` : judge score (surrogate signal)

Goal: Estimate the value of target policy π′:

.. math::

   V(\pi') = E_{x \sim P(X)} E_{a \sim \pi'(\cdot|x)} [R(x, a)]

Core Assumptions
----------------

Data Assumptions (D1-D3)
~~~~~~~~~~~~~~~~~~~~~~~~~

**D1. IID Sampling**: Data is sampled i.i.d. from the logging policy:

.. math::

   (x_i, a_i) \sim P(X) \times \pi_0(\cdot|X)

**D2. Overlap**: The target policy has support contained in the logging policy:

.. math::

   \pi'(a|x) > 0 \implies \pi_0(a|x) > 0

**D3. Bounded Importance Weights**: There exists M < ∞ such that:

.. math::

   \sup_{x,a} \frac{\pi'(a|x)}{\pi_0(a|x)} \leq M

Judge Assumptions (J1-J2)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**J1. Monotone Sufficiency**: The judge score S is monotonically sufficient for the reward:

.. math::

   E[R|S=s] \text{ is monotone in } s

**J2. Oracle Informativeness**: Oracle labels provide unbiased information about true rewards:

.. math::

   E[Y|S=s] = E[R|S=s]

Where Y denotes oracle labels on the calibration slice.

Regularity Assumptions (R1-R3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**R1. Moment Conditions**: Rewards and weights have finite second moments:

.. math::

   E[R^2] < \infty, \quad E[w^2] < \infty

**R2. Nuisance Rate Conditions**: For doubly robust estimation, nuisance functions converge at rate n^(-1/4) or faster:

.. math::

   ||\hat{g} - g^*||_2 = O_p(n^{-1/4})
   ||\hat{w} - w^*||_2 = O_p(n^{-1/4})

**R3. Donsker Conditions**: The function classes for g and w are Donsker.

Main Theoretical Results
------------------------

Theorem 1: Mean Preservation Under Isotonic Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statement**: Let f be the isotonic regression of Y on S. Then:

.. math::

   E[f(S)] = E[Y]

**Proof**: By the characterization of isotonic regression as the L₂ projection onto monotone functions:

.. math::

   f = \arg\min_{g \in \mathcal{M}} E[(Y - g(S))^2]

Where :math:`\mathcal{M}` is the space of monotone functions. The first-order condition gives:

.. math::

   E[(Y - f(S))h(S)] = 0 \quad \forall h \in \mathcal{M}

Taking h(s) = 1 (constant function, which is monotone):

.. math::

   E[Y - f(S)] = 0 \implies E[f(S)] = E[Y]

Theorem 2: Variance Dominance of SIMCal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statement**: Let :math:`\tilde{w}` be the SIMCal-calibrated weights with variance cap c. Then:

.. math::

   \text{Var}[\tilde{w}R] \leq \min(c \cdot \text{Var}[R], \text{Var}[wR])

**Proof Sketch**: 

1. SIMCal solves the optimization problem:

   .. math::

      \min_{w' \in \mathcal{M}_S} \text{Var}[w'R] \quad \text{s.t.} \quad E[w'] = 1, \text{Var}[w'] \leq c

2. The constraint set includes w' = 1 (baseline), so:

   .. math::

      \text{Var}[\tilde{w}R] \leq \text{Var}[1 \cdot R] = \text{Var}[R]

3. By monotonicity and the variance constraint:

   .. math::

      \text{Var}[\tilde{w}R] \leq c \cdot \text{Var}[R]

4. If raw weights w are feasible (satisfy constraints), then:

   .. math::

      \text{Var}[\tilde{w}R] \leq \text{Var}[wR]

Theorem 3: √n-Consistency of Cal-IPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statement**: Under assumptions D1-D3, J1-J2, R1, the Cal-IPS estimator satisfies:

.. math::

   \sqrt{n}(\hat{V}_{Cal-IPS} - V(\pi')) \xrightarrow{d} N(0, \sigma^2)

Where:

.. math::

   \sigma^2 = \text{Var}[\tilde{w}(R - V(\pi'))]

**Proof Outline**:

1. Express the estimator as:

   .. math::

      \hat{V}_{Cal-IPS} = \frac{1}{n}\sum_{i=1}^n \tilde{w}_i R_i

2. By mean preservation: :math:`E[\tilde{w}_i R_i] = V(\pi')`

3. Apply CLT to the centered sum:

   .. math::

      \sqrt{n}(\hat{V}_{Cal-IPS} - V(\pi')) = \frac{1}{\sqrt{n}}\sum_{i=1}^n (\tilde{w}_i R_i - V(\pi'))

4. Verify Lindeberg conditions using bounded weights from SIMCal

5. Conclude asymptotic normality with variance :math:`\sigma^2`

Theorem 4: Double Robustness of DR-CPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statement**: The DR-CPO estimator is consistent if either:
- The outcome model g is consistent, OR
- The importance weights w are consistent

**Proof**: The DR-CPO estimator has the form:

.. math::

   \hat{V}_{DR} = \frac{1}{n}\sum_{i=1}^n \left[\hat{g}(x_i, a_i) + \hat{w}_i(R_i - \hat{g}(x_i, a_i))\right]

Taking expectations:

.. math::

   E[\hat{V}_{DR}] = E[\hat{g}(X, A)] + E[\hat{w}(R - \hat{g}(X, A))]

Case 1: If :math:`\hat{g} \to g^* = E[R|X, A]`:

.. math::

   E[\hat{V}_{DR}] \to E[g^*(X, A)] + E[\hat{w}(R - g^*(X, A))] = V(\pi')

Since :math:`E[R - g^*(X, A)|X, A] = 0`.

Case 2: If :math:`\hat{w} \to w^* = \pi'(A|X)/\pi_0(A|X)`:

.. math::

   E[\hat{V}_{DR}] \to E[\hat{g}(X, A)] + E[w^*(R - \hat{g}(X, A))]
   
   = E[\hat{g}(X, A)] + V(\pi') - E[w^*\hat{g}(X, A)]
   
   = V(\pi')

Since :math:`E[w^*\hat{g}(X, A)] = E[\hat{g}(X, A)]` by importance weighting.

Theorem 5: Semiparametric Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statement**: Under regularity conditions, DR-CPO achieves the semiparametric efficiency bound:

.. math::

   \sqrt{n}(\hat{V}_{DR} - V(\pi')) \xrightarrow{d} N(0, \sigma^2_{eff})

Where :math:`\sigma^2_{eff}` is the variance of the efficient influence function:

.. math::

   \psi_{eff}(x, a, r) = g^*(x, a) + w^*(x, a)(r - g^*(x, a)) - V(\pi')

**Key Insight**: This is the smallest possible asymptotic variance among all regular estimators.

Practical Implications
----------------------

Bias-Variance Tradeoff
~~~~~~~~~~~~~~~~~~~~~~~

The choice of variance cap in SIMCal controls the bias-variance tradeoff:

.. list-table::
   :header-rows: 1

   * - Variance Cap
     - Bias
     - Variance
     - MSE
   * - Small (0.5)
     - Higher
     - Lower
     - Good for small n
   * - Medium (1.0)
     - Moderate
     - Moderate
     - Balanced
   * - Large (2.0)
     - Lower
     - Higher
     - Good for large n
   * - None
     - Zero (asymptotic)
     - Highest
     - Good for very large n

Sample Size Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

Rough guidelines based on theory and empirics:

.. list-table::
   :header-rows: 1

   * - Estimator
     - Minimum n
     - Recommended n
     - Notes
   * - RawIPS
     - 10,000
     - 50,000+
     - Needs large n for stability
   * - CalibratedIPS
     - 100
     - 1,000+
     - SIMCal enables smaller samples
   * - DR-CPO
     - 500
     - 2,000+
     - Needs fresh draws
   * - TMLE
     - 1,000
     - 5,000+
     - Most complex, best MSE

Oracle Coverage Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For effective calibration:

- **Minimum**: 5% of data with oracle labels
- **Recommended**: 10-20% for robust calibration
- **Diminishing returns**: >30% provides marginal benefit

The oracle labels should span the range of judge scores for good interpolation.

Convergence Rates
~~~~~~~~~~~~~~~~~

Under standard conditions:

.. list-table::
   :header-rows: 1

   * - Component
     - Rate
     - Notes
   * - Cal-IPS
     - :math:`O_p(n^{-1/2})`
     - Standard √n rate
   * - Isotonic calibration
     - :math:`O_p(n^{-2/3})`
     - Cube-root rate for isotonic
   * - DR-CPO
     - :math:`O_p(n^{-1/2})`
     - √n if either nuisance is n^(-1/4)
   * - SIMCal projection
     - :math:`O_p(n^{-1/2})`
     - Inherits parametric rate

Extensions and Future Work
--------------------------

Potential theoretical extensions:

1. **Adaptive variance caps**: Data-driven selection of optimal cap
2. **Higher-order calibrations**: Beyond monotone to smooth functions
3. **Multi-dimensional surrogates**: Using multiple judge signals
4. **Online/streaming**: Sequential updates for production systems
5. **Finite-sample bounds**: PAC-style guarantees

References
----------

Key papers for theoretical foundations:

- Horvitz & Thompson (1952): Basic importance sampling
- van der Laan et al. (2025): Isotonic calibration theory
- Dudík et al. (2011): Doubly robust policy evaluation
- Bickel et al. (1993): Semiparametric efficiency theory
- Chernozhukov et al. (2018): Double/debiased machine learning

See the forthcoming paper for complete proofs and additional results.
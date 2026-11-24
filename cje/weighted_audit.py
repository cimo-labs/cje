"""Design-based paired audits of a known, finite frame of independent clusters."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class WeightedPairedAuditResult:
    """A fixed-frame estimate in the supplied units, and its sampling diagnostics.

    `estimate` is the untruncated Horvitz–Thompson difference estimator.
    `ci` is a conservative, fixed-look design confidence interval.
    `outcome_identification_bounds` uses only the observed human outcomes,
    known frame masses and declared outcome support; it always contains truth.
    `traffic_effective_n` and `label_effective_n` describe the concentration
    of the realized cluster weights. Neither is an independent sample count.
    """

    estimate: float
    ci: tuple[float, float]
    outcome_identification_bounds: tuple[float, float]
    alpha: float
    n_frame: int
    n_traffic: int
    n_labeled: int
    traffic_effective_n: float
    label_effective_n: float
    labeled_target_mass: float
    min_joint_inclusion: float


def _frame_vector(values: ArrayLike, n: int, name: str) -> np.ndarray:
    if np.ma.is_masked(values):
        raise ValueError(f"{name} has masked entries; use explicit masks and NaN")
    arr = np.asarray(values)
    if arr.shape != (n,):
        raise ValueError(f"{name} must contain one value per frame cluster")
    return arr


def _probabilities(values: ArrayLike, n: int, name: str) -> np.ndarray:
    arr = _frame_vector(values, n, name)
    if arr.dtype.kind not in "ifu":
        raise ValueError(f"{name} must contain numeric inclusion probabilities")
    prob = arr.astype(float)
    if not np.all(np.isfinite(prob) & (prob > 0) & (prob <= 1)):
        raise ValueError(f"{name} must be finite and in (0, 1] for every frame cluster")
    return prob


def _observations(
    values: ArrayLike, mask: np.ndarray, name: str, lower: float, upper: float
) -> np.ndarray:
    arr = _frame_vector(values, len(mask), name)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real")
    if arr.dtype.kind in "US" or (
        arr.dtype.kind == "O" and any(isinstance(v, (str, bytes)) for v in arr)
    ):
        raise ValueError(f"{name} must be numeric (NaN or None for missing), not text")
    arr = arr.astype(float)
    if not np.all(np.isnan(arr[~mask])):
        raise ValueError(
            f"{name} must be missing (NaN or None) outside its sampling mask"
        )
    observed = arr[mask]
    if not np.all(np.isfinite(observed) & (observed >= lower) & (observed <= upper)):
        raise ValueError(f"selected {name} must be finite and inside outcome_bounds")
    return np.asarray(observed)


def _kish(weights: np.ndarray) -> float:
    if not weights.size:
        return 0.0
    scaled = weights / weights.max()
    return float(scaled.sum() ** 2 / (scaled @ scaled))


def weighted_paired_audit(
    *,
    cluster_ids: ArrayLike,
    traffic_selected: ArrayLike,
    label_selected: ArrayLike,
    traffic_inclusion: ArrayLike,
    label_inclusion: ArrayLike,
    predictions: ArrayLike,
    outcomes: ArrayLike,
    outcome_bounds: tuple[float, float],
    target_weights: ArrayLike | None = None,
    alpha: float = 0.05,
) -> WeightedPairedAuditResult:
    """Estimate a prespecified paired outcome on a known, fixed cluster frame.

    Supply a row for *every* independent cluster, including unsampled ones.
    Collapse repeated draws and both arms to one prespecified scalar contrast
    (or paired loss) per cluster *before* this call. All arrays must be aligned.
    `traffic_inclusion` is the known first-stage Bernoulli probability q.
    `label_inclusion` is the known conditional second-stage probability p
    given traffic selection. Both must be known for every row. Selections are
    independent across clusters. Within a cluster both arms are selected
    together. Fixed quotas, estimated propensities and adaptive stopping have
    different design distributions.

    `traffic_selected` and `label_selected` are boolean design indicators.
    `predictions` holds a fixed bounded auxiliary prediction on selected
    traffic rows. `outcomes` contains the actual paired outcome on labeled
    rows. Use NaN/None elsewhere. A predictor trained on a separate historical
    sample is allowed; condition on that fit. The prediction and conditional
    label probabilities must be fixed independently of the random audit draws.

    `outcome_bounds` are prespecified finite lower/upper endpoints for every
    prediction and outcome, including unobserved values. For a difference of
    two [0,1] outcomes these can be (-1,1); for a new-failure indicator (0,1).
    `target_weights` are optional known positive cluster masses over the full
    frame, normalized to sum to one. Omit for an equal-cluster target.

    The point estimate uses traffic-weighted predictions and a joint-inclusion
    weighted correction. The fixed-look confidence interval uses Bernstein's
    inequality and the declared outcome bounds; its width does not depend on the
    predictions or outcomes. No human outcomes yields the full declared
    interval. See guides/weighted-paired-audit.md for the proof, supported
    designs, and an example using historical calibration weights.
    """
    ids = np.asarray(cluster_ids, dtype=object)
    if ids.ndim != 1 or not ids.size or np.ma.is_masked(cluster_ids):
        raise ValueError("cluster_ids must identify a nonempty one-dimensional frame")
    if any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (str, Integral))
        or (isinstance(i, str) and not i.strip())
        for i in ids
    ):
        raise ValueError("cluster_ids must be nonempty strings or integers")
    if any(isinstance(i, str) for i in ids) and any(
        isinstance(i, Integral) for i in ids
    ):
        raise ValueError(
            "cluster_ids mix strings and integers; use one type so 1 and '1' "
            "cannot name different clusters"
        )
    n = len(ids)
    if len(set(ids)) != n:
        raise ValueError(
            "cluster_ids must be unique; aggregate repeated draws by cluster"
        )
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, Real)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be a probability strictly between zero and one")
    bounds = np.asarray(outcome_bounds)
    if bounds.shape != (2,) or bounds.dtype.kind not in "ifu":
        raise ValueError("outcome_bounds must contain two numeric endpoints")
    lower, upper = map(float, bounds)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError("outcome_bounds must be finite and strictly increasing")

    q = _probabilities(traffic_inclusion, n, "traffic_inclusion")
    p = _probabilities(label_inclusion, n, "label_inclusion")
    seen = _frame_vector(traffic_selected, n, "traffic_selected")
    rated = _frame_vector(label_selected, n, "label_selected")
    if seen.dtype != np.bool_ or rated.dtype != np.bool_:
        raise ValueError("selection masks must be boolean")
    if np.any(rated & ~seen):
        raise ValueError("a labeled cluster must be selected in traffic")
    if np.any((q == 1) & ~seen) or np.any(seen & (p == 1) & ~rated):
        raise ValueError(
            "a probability-one draw cannot be missing from its sampling mask"
        )
    prediction = _observations(predictions, seen, "predictions", lower, upper)
    outcome = _observations(outcomes, rated, "outcomes", lower, upper)

    if target_weights is None:
        w = np.full(n, 1 / n)
    else:
        masses = _frame_vector(target_weights, n, "target_weights")
        if masses.dtype.kind not in "ifu" or not np.all(
            np.isfinite(masses) & (masses > 0)
        ):
            raise ValueError(
                "target_weights must be finite, positive, known frame masses"
            )
        masses = masses.astype(float) / masses.max()
        w = masses / masses.sum()
        if np.any(w == 0):
            raise ValueError("target_weights exceed the numerical range")

    # Scale the outcome units before squaring or computing a radius. Dividing
    # both endpoints by the same positive number preserves the raw HT estimand.
    scale = max(abs(lower), abs(upper))
    lo, hi = lower / scale, upper / scale
    prediction, outcome = prediction / scale, outcome / scale
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        joint = q * p
        weight_traffic = w / q
        weight_label = w / joint
        if not (np.all(joint > 0) and np.all(np.isfinite(weight_label))):
            raise ValueError("joint inclusion probabilities exceed the numerical range")
        rated_among_seen = rated[seen]
        estimate = float(
            weight_traffic[seen & ~rated] @ prediction[~rated_among_seen]
            + weight_traffic[rated] @ outcome
            + (weight_label[rated] * (1 - p[rated]))
            @ (outcome - prediction[rated_among_seen])
        )
        # Var(X) = (1-q)/q*y² + (1-p)/(q*p)*(y-m)². The variance and
        # maximum increment used for concentration must be fixed BEFORE draws.
        # Here max |y|=1 in scaled units and |y-m|<=hi-lo.
        span = hi - lo
        first = weight_traffic * (1 - q)
        second = weight_label * (1 - p)
        variance_bound = float(w @ (first + second * span**2))
        # The three possible |w*(X-y)| bounds correspond to no traffic,
        # traffic without a label, and selection at both stages.
        no_traffic = np.where(q < 1, w, 0.0)
        no_label = np.where(p < 1, first + weight_traffic * span, 0.0)
        labeled = first + second * span
        max_error = float(np.max(np.maximum(np.maximum(no_traffic, no_label), labeled)))
        log_tail = np.log(2.0) - np.log(alpha)
        linear = max_error * log_tail / 3.0
        radius = linear + np.hypot(np.sqrt(2.0 * variance_bound * log_tail), linear)
        observed_total = float(w[rated] @ outcome)
        unseen_mass = float(w[~rated].sum())
        support_lower = max(lo, observed_total + unseen_mass * lo)
        support_upper = min(hi, observed_total + unseen_mass * hi)
        # Roundoff near an endpoint may put the observed total one ulp outside.
        support_lower = min(support_lower, hi)
        support_upper = max(support_lower, support_upper)
        if not np.any(rated):
            support_lower, support_upper = lo, hi
        left = max(support_lower, estimate - radius)
        right = min(support_upper, estimate + radius)
        if left > right or not np.any(rated):
            left, right = support_lower, support_upper
        estimate *= scale

    if not np.isfinite(estimate):
        raise ValueError("weighted estimate exceeds the numerical range")
    return WeightedPairedAuditResult(
        estimate=estimate,
        ci=(float(left * scale), float(right * scale)),
        outcome_identification_bounds=(support_lower * scale, support_upper * scale),
        alpha=float(alpha),
        n_frame=n,
        n_traffic=int(seen.sum()),
        n_labeled=int(rated.sum()),
        traffic_effective_n=_kish(weight_traffic[seen]),
        label_effective_n=_kish(weight_label[rated]),
        labeled_target_mass=float(w[rated].sum()),
        min_joint_inclusion=float(joint.min()),
    )

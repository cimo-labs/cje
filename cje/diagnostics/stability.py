"""
Stability and drift detection diagnostics for judges and models.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def kendall_tau_drift(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    labels_1: Optional[np.ndarray] = None,
    labels_2: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute Kendall τ rank correlation for drift detection.

    Detects changes in judge ranking behavior over time/batches.

    Args:
        scores_1: Judge scores from period/batch 1
        scores_2: Judge scores from period/batch 2
        labels_1: Optional labels for period 1 (for paired comparison)
        labels_2: Optional labels for period 2 (for paired comparison)

    Returns:
        Dictionary with:
        - 'tau': Kendall τ statistic (-1 to 1)
        - 'p_value': Significance of difference from perfect correlation
        - 'drift_detected': Boolean (True if p < 0.05)
        - 'interpretation': String description

    References:
        Section 9.5 of the CJE paper on temporal stability.
    """
    # Basic Kendall τ between two score sets
    if len(scores_1) != len(scores_2):
        raise ValueError(
            f"Score arrays must have same length: {len(scores_1)} vs {len(scores_2)}"
        )

    tau, p_value = stats.kendalltau(scores_1, scores_2)

    # Detect significant drift (handle NaN)
    drift_detected = False
    if not np.isnan(tau) and not np.isnan(p_value):
        drift_detected = p_value < 0.05 and tau < 0.8

    # Interpretation (handle NaN)
    if np.isnan(tau):
        interpretation = "Cannot compute stability (constant values)"
    elif tau > 0.9:
        interpretation = "Excellent stability (τ > 0.9)"
    elif tau > 0.8:
        interpretation = "Good stability (0.8 < τ ≤ 0.9)"
    elif tau > 0.6:
        interpretation = "Moderate drift detected (0.6 < τ ≤ 0.8)"
    elif tau > 0.4:
        interpretation = "Significant drift detected (0.4 < τ ≤ 0.6)"
    else:
        interpretation = "Severe drift detected (τ ≤ 0.4)"

    result = {
        "tau": float(tau),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected),
        "interpretation": interpretation,
        "n_samples": len(scores_1),
    }

    # If labels provided, compute drift in score-label relationship
    if labels_1 is not None and labels_2 is not None:
        tau1, _ = stats.kendalltau(scores_1, labels_1)
        tau2, _ = stats.kendalltau(scores_2, labels_2)

        result["tau_with_labels_1"] = float(tau1)
        result["tau_with_labels_2"] = float(tau2)
        result["tau_change"] = float(tau2 - tau1)

        if abs(tau2 - tau1) > 0.1:
            result["calibration_drift"] = True
            result["calibration_drift_severity"] = (
                "high" if abs(tau2 - tau1) > 0.2 else "moderate"
            )

    return result


def sequential_drift_detection(
    score_batches: List[np.ndarray],
    window_size: int = 2,
    overlap: int = 1,
) -> Dict[str, Any]:
    """Detect drift across multiple sequential batches.

    Args:
        score_batches: List of score arrays for each time period
        window_size: Number of batches to compare
        overlap: Number of overlapping batches between windows

    Returns:
        Dictionary with:
        - 'tau_sequence': List of τ values between consecutive batches
        - 'drift_points': Indices where drift detected
        - 'overall_stability': Summary metric
    """
    if len(score_batches) < 2:
        return {
            "tau_sequence": [],
            "drift_points": [],
            "overall_stability": 1.0,
            "insufficient_data": True,
        }

    tau_sequence = []
    drift_points = []

    # Compare consecutive batches
    for i in range(len(score_batches) - 1):
        # Need same size for comparison
        min_len = min(len(score_batches[i]), len(score_batches[i + 1]))
        if min_len < 10:
            continue

        scores1 = score_batches[i][:min_len]
        scores2 = score_batches[i + 1][:min_len]

        result = kendall_tau_drift(scores1, scores2)
        tau_sequence.append(result["tau"])

        if result["drift_detected"]:
            drift_points.append(i + 1)

    # Overall stability as minimum tau (handle NaN and None values)
    if tau_sequence:
        # Filter out NaN and None values before taking min
        valid_taus = [t for t in tau_sequence if t is not None and not np.isnan(t)]
        overall_stability = min(valid_taus) if valid_taus else np.nan
    else:
        overall_stability = 1.0

    return {
        "tau_sequence": tau_sequence,
        "drift_points": drift_points,
        "overall_stability": float(overall_stability),
        "n_batches": len(score_batches),
        "has_drift": len(drift_points) > 0,
    }


def reliability_diagram(
    predicted_probs: np.ndarray,
    true_binary: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute reliability diagram statistics and Brier score decomposition.

    For assessing calibration quality of probabilistic predictions.

    Args:
        predicted_probs: Predicted probabilities [0, 1]
        true_binary: True binary outcomes {0, 1}
        n_bins: Number of bins for reliability diagram

    Returns:
        Dictionary with:
        - 'bin_edges': Bin boundaries
        - 'bin_frequencies': Fraction of predictions in each bin
        - 'bin_accuracies': Actual positive rate in each bin
        - 'bin_confidences': Mean predicted probability in each bin
        - 'ece': Expected Calibration Error
        - 'mce': Maximum Calibration Error
        - 'brier_score': Overall Brier score
        - 'brier_reliability': Reliability component
        - 'brier_resolution': Resolution component
        - 'brier_uncertainty': Uncertainty component

    References:
        Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """
    n = len(predicted_probs)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bin_edges[1:-1])

    bin_frequencies: List[float] = []
    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = mask.sum()

        if n_in_bin > 0:
            bin_frequencies.append(n_in_bin / n)
            bin_accuracies.append(true_binary[mask].mean())
            bin_confidences.append(predicted_probs[mask].mean())
        else:
            bin_frequencies.append(0.0)
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    bin_frequencies_arr = np.array(bin_frequencies)
    bin_accuracies_arr = np.array(bin_accuracies)
    bin_confidences_arr = np.array(bin_confidences)

    # Expected Calibration Error (ECE)
    ece = np.sum(bin_frequencies_arr * np.abs(bin_accuracies_arr - bin_confidences_arr))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(bin_accuracies_arr - bin_confidences_arr))

    # Brier Score and decomposition
    brier_score = np.mean((predicted_probs - true_binary) ** 2)

    # Decomposition: BS = Reliability - Resolution + Uncertainty
    base_rate = true_binary.mean()
    uncertainty = base_rate * (1 - base_rate)

    # Reliability: weighted squared difference between confidence and accuracy
    reliability = np.sum(
        bin_frequencies_arr * (bin_confidences_arr - bin_accuracies_arr) ** 2
    )

    # Resolution: how much the predictions vary from base rate
    resolution = np.sum(bin_frequencies_arr * (bin_accuracies_arr - base_rate) ** 2)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_frequencies": bin_frequencies_arr.tolist(),
        "bin_accuracies": bin_accuracies_arr.tolist(),
        "bin_confidences": bin_confidences_arr.tolist(),
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier_score),
        "brier_reliability": float(reliability),
        "brier_resolution": float(resolution),
        "brier_uncertainty": float(uncertainty),
        "is_calibrated": ece < 0.1,  # Rule of thumb threshold
    }


def eif_qq_plot_data(
    influence_functions: np.ndarray,
    standardize: bool = True,
) -> Dict[str, Any]:
    """Generate data for EIF Q-Q plot to check normality assumptions.

    Args:
        influence_functions: Array of influence function values
        standardize: Whether to standardize before comparison

    Returns:
        Dictionary with:
        - 'theoretical_quantiles': Expected normal quantiles
        - 'sample_quantiles': Observed quantiles
        - 'shapiro_stat': Shapiro-Wilk test statistic
        - 'shapiro_p': Shapiro-Wilk p-value
        - 'is_normal': Boolean (p > 0.05)
        - 'skewness': Skewness of distribution
        - 'kurtosis': Excess kurtosis
        - 'outlier_indices': Indices of potential outliers
    """
    n = len(influence_functions)

    # Standardize if requested
    if standardize:
        ifs = (
            influence_functions - influence_functions.mean()
        ) / influence_functions.std()
    else:
        ifs = influence_functions.copy()

    # Q-Q plot data
    sample_quantiles = np.sort(ifs)
    theoretical_quantiles = stats.norm.ppf((np.arange(n) + 0.5) / n)

    # Shapiro-Wilk test for normality (works up to n=5000)
    if n <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(ifs)
    else:
        # Use Anderson-Darling for large samples
        ad_result = stats.anderson(ifs, dist="norm")
        shapiro_stat = ad_result.statistic
        # Approximate p-value from critical values
        shapiro_p = 0.01 if ad_result.statistic > ad_result.critical_values[-1] else 0.5

    # Skewness and kurtosis
    skewness = stats.skew(ifs)
    kurtosis = stats.kurtosis(ifs)  # Excess kurtosis (0 for normal)

    # Detect outliers (> 3 std from mean in standardized data)
    if standardize:
        outlier_mask = np.abs(ifs) > 3
    else:
        z_scores = np.abs((ifs - ifs.mean()) / ifs.std())
        outlier_mask = z_scores > 3

    outlier_indices = np.where(outlier_mask)[0].tolist()

    return {
        "theoretical_quantiles": theoretical_quantiles.tolist(),
        "sample_quantiles": sample_quantiles.tolist(),
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "is_normal": shapiro_p > 0.05,
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "n_outliers": len(outlier_indices),
        "outlier_fraction": len(outlier_indices) / n,
        "outlier_indices": outlier_indices[:10],  # Limit to first 10
    }


def compute_stability_diagnostics(
    dataset: Any,
    batch_size: Optional[int] = None,
    judge_field: str = "judge_score",
    oracle_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute comprehensive stability diagnostics for a dataset.

    Args:
        dataset: CJE dataset with samples
        batch_size: Size of batches for drift detection (auto if None)
        judge_field: Field name for judge scores
        oracle_field: Optional field name for oracle labels

    Returns:
        Dictionary with stability metrics
    """
    # Extract judge scores and oracle labels separately
    all_judge_scores = []
    oracle_labels = []
    paired_judge_scores = []

    for sample in dataset.samples:
        if judge_field in sample.metadata:
            judge_score = sample.metadata[judge_field]
            all_judge_scores.append(judge_score)

            # Collect oracle labels separately for calibration checking
            if oracle_field and oracle_field in sample.metadata:
                oracle_label = sample.metadata[oracle_field]
                # Only include non-None oracle labels
                if oracle_label is not None:
                    oracle_labels.append(oracle_label)
                    paired_judge_scores.append(judge_score)

    # Use all judge scores for drift detection
    judge_scores = np.array(all_judge_scores)
    n = len(judge_scores)

    if n == 0:
        return {"error": "No judge scores found"}

    # Determine batch size
    if batch_size is None:
        batch_size = max(50, n // 10)  # At least 50, or 10% of data

    # Create batches for drift detection
    n_batches = n // batch_size
    score_batches = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        score_batches.append(judge_scores[start:end])

    # Sequential drift detection
    drift_result = sequential_drift_detection(score_batches)

    result = {
        "n_samples": n,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "drift_detection": drift_result,
    }

    # If we have oracle labels, check calibration stability
    if len(oracle_labels) > 0:
        oracle_labels_array = np.array(oracle_labels)
        paired_judge_scores_array = np.array(paired_judge_scores)

        # Overall correlation (using only paired data)
        overall_tau, _ = stats.kendalltau(
            paired_judge_scores_array, oracle_labels_array
        )
        # Handle potential None or NaN from kendalltau
        if overall_tau is None:
            result["overall_tau_with_oracle"] = np.nan
        else:
            result["overall_tau_with_oracle"] = float(overall_tau)

        # Check correlation stability across batches (using paired data)
        tau_per_batch = []
        n_paired = len(paired_judge_scores_array)
        paired_batch_size = max(
            10, n_paired // 10
        )  # At least 10, or 10% of paired data
        n_paired_batches = n_paired // paired_batch_size

        for i in range(n_paired_batches):
            start = i * paired_batch_size
            end = min((i + 1) * paired_batch_size, n_paired)

            if end - start >= 2:  # Need at least 2 points for correlation
                batch_tau, _ = stats.kendalltau(
                    paired_judge_scores_array[start:end], oracle_labels_array[start:end]
                )
                # Ensure we handle potential None values from kendalltau
                if batch_tau is None:
                    tau_per_batch.append(np.nan)
                else:
                    tau_per_batch.append(batch_tau)

        result["tau_with_oracle_per_batch"] = tau_per_batch
        # Handle NaN and None values in tau_per_batch when computing std
        valid_taus = [t for t in tau_per_batch if t is not None and not np.isnan(t)]
        if valid_taus:
            result["tau_stability"] = float(np.std(valid_taus))
        else:
            result["tau_stability"] = np.nan

    return result

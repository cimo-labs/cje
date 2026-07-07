"""
High-level analysis functions for CJE.

This module provides simple, one-line analysis functions that handle
the complete CJE workflow automatically. The whole Direct-mode pipeline
lives here: validate inputs -> load draws -> resolve calibration source ->
calibrate -> CalibratedDirectEstimator -> estimate -> denormalize ->
metadata.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..calibration import calibrate_dataset
from ..data import load_dataset_from_jsonl
from ..data.models import Dataset, EstimationResult
from ._removed import validate_estimator_name

logger = logging.getLogger(__name__)

# Exact migration copy for the removed OPE (IPS/DR) entry point. Single source
# for the API and the CLI; pinned verbatim by test_migration_errors.py.
LOGGED_DATA_PATH_REMOVED_MESSAGE = """\
Off-policy evaluation was removed in cje-eval 0.4.0; 'logged_data_path' is no longer accepted.

CJE is now Direct-mode only: generate fresh draws from each policy, judge them,
and label a small oracle slice.

  * Have fresh draws?  Pass fresh_draws_dir=... or fresh_draws_data=... .
  * Your logged data has judge_score + oracle_label?  It still works as the
    calibration source: pass calibration_data_path="<your logged data>.jsonl".
  * Need IPS/DR from logged propensities?  Pin the frozen OPE line:
        pip install "cje-eval==0.3.*"
    (maintained on the 0.3.x branch; docs at the v0.3.0 tag; requires
    Python <=3.12 — on 3.13 use a 3.12 env for OPE)."""

# Keys analyze_dataset accepts in estimator_config. `oua_jackknife` is popped
# and merged with the pipeline's default; the rest are forwarded verbatim to
# CalibratedDirectEstimator. `reward_calibrator` is deliberately absent: the
# pipeline manages it.
_ESTIMATOR_CONFIG_KEYS = (
    "oua_jackknife",
    "inference_method",
    "n_bootstrap",
    "bootstrap_seed",
    "use_augmented_estimator",
    "paired_comparison",
)


def analyze_dataset(
    logged_data_path: Optional[str] = None,
    fresh_draws_dir: Optional[str] = None,
    fresh_draws_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    calibration_data_path: Optional[str] = None,
    combine_oracle_sources: bool = True,
    estimator: str = "auto",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    calibration_covariates: Optional[List[str]] = None,
    include_response_length: bool = False,
    estimator_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> EstimationResult:
    """
    Analyze policies using fresh draws (Direct mode).

    This high-level function handles:
    - Data loading and validation
    - Automatic reward calibration (judge → oracle mapping)
    - Oracle source combining (pooling labels from multiple sources)
    - Complete analysis workflow

    Args:
        logged_data_path: REMOVED in 0.4.0 along with the OPE (IPS/DR) modes.
            Accepted only so that passing a value raises a ValueError with the
            migration guidance. Logged data with judge scores and oracle labels
            still works as the calibration source via calibration_data_path.
        fresh_draws_dir: Directory containing per-policy fresh draw response
            files, or a single JSONL file whose records carry a target_policy
            field. Judge scores on any bounded scale (0-100, Likert 1-5, ...)
            are auto-normalized, exactly as with fresh_draws_data.
        fresh_draws_data: In-memory alternative to fresh_draws_dir. Dict mapping policy names
            to lists of records. Each record needs: prompt_id, judge_score. Optional: oracle_label.
            Example: {"policy_a": [{"prompt_id": "1", "judge_score": 0.8}, ...], ...}
        calibration_data_path: Path to dedicated calibration dataset with oracle labels.
            Use this to learn judge→oracle mapping from a curated oracle set separate
            from your evaluation data. If combine_oracle_sources=True (default), will
            pool with oracle labels from fresh_draws for maximum efficiency.
        combine_oracle_sources: Whether to pool oracle labels from all sources
            (calibration_data + fresh_draws). Default True for data efficiency.
            Set False to use ONLY calibration_data_path for learning calibration.
            When combining, every source's (judge, oracle) pair enters calibration
            (labels attach to responses); only true duplicates are deduped.
        estimator: Estimator type. Options:
            - "auto" (default): resolves to "direct"
            - "direct" / "calibrated-direct": On-policy evaluation (requires fresh_draws_dir
              or fresh_draws_data)
            Removed OPE names (calibrated-ips, raw-ips, dr-cpo, mrdr, tmle,
            stacked-dr) raise a ValueError pointing at the 0.3.x releases.
        judge_field: Metadata field containing judge scores (default "judge_score")
        oracle_field: Metadata field containing oracle labels (default "oracle_label")
        calibration_covariates: Optional list of metadata field names to use as covariates
            in two-stage reward calibration (e.g., ["response_length", "domain"]).
            Helps handle confounding where judge scores at fixed S have different oracle
            outcomes based on observable features like response length or domain.
            Only works with two_stage or auto calibration mode.
        include_response_length: Automatically include response length (word count) as a covariate.
            Computed as len(response.split()). Requires all samples (fresh draws
            and calibration data) to have a 'response' field. If True, 'response_length' is
            automatically prepended to calibration_covariates. Convenient for handling length bias.
        estimator_config: Optional configuration dict for the estimator. Valid
            keys: oua_jackknife, inference_method, n_bootstrap, bootstrap_seed,
            use_augmented_estimator, paired_comparison. Unknown keys raise a
            ValueError; reward_calibrator is managed by analyze_dataset and is
            rejected.
        verbose: Whether to print progress messages

    Returns:
        EstimationResult with estimates, standard errors, and metadata.

        New metadata fields when using calibration_data_path:
        - results.metadata["oracle_sources"]: Breakdown of oracle labels by source
        - results.metadata["oracle_sources"]["conflicts"]: Cross-source oracle disagreements

    Raises:
        ValueError: If required data is missing for the selected estimator

    Example - Basic usage:
        >>> # Direct mode: Fresh draws only
        >>> results = analyze_dataset(fresh_draws_dir="responses/")

    Example - Dedicated calibration set:
        >>> # Learn calibration from curated oracle set
        >>> results = analyze_dataset(
        ...     fresh_draws_dir="responses/",
        ...     calibration_data_path="human_labels.jsonl",  # 1000 samples, high quality
        ... )
        >>> print(f"Oracle sources: {results.metadata['oracle_sources']}")
    """
    from ..data.fresh_draws import (
        NormalizationInfo,
        compute_response_covariates,
        fresh_draws_data_from_dir,
        fresh_draws_from_dict,
    )
    from ..data.ingest import fresh_draws_data_from_file
    from ..data.models import Sample
    from ..estimators.direct_method import CalibratedDirectEstimator

    # --- 1. Migration check (OPE removed in 0.4.0)
    if logged_data_path is not None:
        raise ValueError(LOGGED_DATA_PATH_REMOVED_MESSAGE)

    # --- 2. Presence check: at least one fresh-draws source
    if fresh_draws_dir is None and fresh_draws_data is None:
        raise ValueError(
            "Must provide at least one of: fresh_draws_dir, fresh_draws_data"
        )

    # --- 3. Estimator-name validation ("auto" resolves to "direct"; removed
    # 0.3.x OPE names raise the migration error)
    chosen_estimator = (estimator or "auto").strip().lower()
    if chosen_estimator == "auto":
        chosen_estimator = "direct"
    else:
        validate_estimator_name(chosen_estimator)

    # --- 4. Load fresh draws (in-memory dict, directory, or single
    # multi-policy JSONL file). All three inputs converge on
    # fresh_draws_from_dict so joint scale detection / auto-normalization
    # behaves identically; norm_info tracks the applied normalization for
    # the inverse transform in stage 7.
    norm_info: Optional[NormalizationInfo] = None

    if fresh_draws_data is not None:
        if verbose:
            logger.info("Using in-memory fresh_draws_data")
        raw_fresh_draws = fresh_draws_data
    else:
        assert fresh_draws_dir is not None  # validated in stage 2
        draws_path = Path(fresh_draws_dir)
        if draws_path.is_file():
            # Single JSONL file with a target_policy field per record
            if verbose:
                logger.info(f"Loading fresh draws from file {draws_path}")
            raw_fresh_draws = fresh_draws_data_from_file(draws_path)
        else:
            raw_fresh_draws = fresh_draws_data_from_dir(draws_path, verbose=verbose)

    fresh_draws_dict, norm_info = fresh_draws_from_dict(
        raw_fresh_draws, verbose=verbose, auto_normalize=True
    )
    target_policies = sorted(fresh_draws_dict.keys())

    if verbose:
        logger.info(
            f"Found {len(target_policies)} policies: {', '.join(target_policies)}"
        )

    all_fresh_draws = []
    for fd in fresh_draws_dict.values():
        all_fresh_draws.extend(fd.samples)

    # --- 5. Resolve the calibration source and calibrate:
    # calibration_data_path (optionally pooled with labeled draws via
    # _combine_oracle_sources) OR oracle-in-draws
    oracle_sources_metadata = None
    calibration_result = None
    covariate_names = None
    oracle_coverage = 0.0

    if calibration_data_path:
        if verbose:
            logger.info(f"Loading calibration dataset from {calibration_data_path}")
        calibration_dataset = load_dataset_from_jsonl(calibration_data_path)
        if verbose:
            logger.info(
                f"Loaded calibration dataset: {calibration_dataset.n_samples} samples"
            )

        if combine_oracle_sources:
            if verbose:
                logger.info(
                    "Combining oracle sources from calibration data and fresh draws"
                )
            calibration_dataset_for_rewards, oracle_sources_metadata = (
                _combine_oracle_sources(
                    calibration_dataset,
                    None,  # No logged dataset in Direct mode
                    fresh_draws_dict,
                    target_policies,
                    judge_field,
                    oracle_field,
                    verbose,
                )
            )
        else:
            # Use ONLY calibration_data_path for learning calibration
            if verbose:
                logger.info(
                    "Using calibration data exclusively (combine_oracle_sources=False)"
                )
            calibration_dataset_for_rewards = calibration_dataset
            n_calib_oracle = sum(
                1 for s in calibration_dataset.samples if s.oracle_label is not None
            )
            oracle_sources_metadata = {
                "calibration_data": {
                    "n_oracle": n_calib_oracle,
                    "coverage": n_calib_oracle / calibration_dataset.n_samples,
                },
                "total_oracle": n_calib_oracle,
                "combine_enabled": False,
            }

        # Covariate list (validates response fields when
        # include_response_length is set), then learn the calibration
        # (calibrate_dataset auto-reduces the fold count when labels
        # are scarce)
        covariate_names = _build_covariate_list(
            calibration_covariates,
            include_response_length,
            calibration_dataset_for_rewards,
        )
        _, calibration_result = calibrate_dataset(
            calibration_dataset_for_rewards,
            judge_field=judge_field,
            oracle_field=oracle_field,
            enable_cross_fit=True,
            covariate_names=covariate_names,
        )
    else:
        # Oracle-in-draws: learn calibration from labeled fresh draws
        n_with_oracle = sum(1 for s in all_fresh_draws if s.oracle_label is not None)
        oracle_coverage = n_with_oracle / len(all_fresh_draws) if all_fresh_draws else 0

        if oracle_coverage > 0:
            if verbose:
                logger.info(
                    f"Found {n_with_oracle}/{len(all_fresh_draws)} samples with oracle labels ({oracle_coverage:.1%})"
                )
                logger.info("Learning calibration from fresh draws")

            # Convert FreshDrawSample to Sample so the draws can serve
            # as a calibration dataset
            calibration_samples = []
            for fd_sample in all_fresh_draws:
                sample = Sample(
                    prompt_id=fd_sample.prompt_id,
                    prompt="",  # Not needed for calibration
                    response=fd_sample.response or "",
                    reward=None,  # Will be calibrated
                    judge_score=fd_sample.judge_score,
                    oracle_label=fd_sample.oracle_label,
                    metadata={},
                )
                calibration_samples.append(sample)
            fresh_dataset = Dataset(
                samples=calibration_samples, target_policies=target_policies
            )

            covariate_names = _build_covariate_list(
                calibration_covariates, include_response_length, fresh_dataset
            )
            _, calibration_result = calibrate_dataset(
                fresh_dataset,
                judge_field=judge_field,
                oracle_field=oracle_field,
                enable_cross_fit=True,
                covariate_names=covariate_names,
            )
        else:
            logger.warning(
                "No oracle labels found — returning UNCALIBRATED judge-score "
                "means; CIs do not account for judge bias. Results will be "
                "labeled method='naive_direct'. Add oracle labels to the "
                "fresh draws or provide calibration_data_path to calibrate."
            )

    # --- 6. Build the estimator (validating estimator_config), attach draws
    # (computing covariates when the calibrator expects them), and estimate
    oua_jackknife, estimator_kwargs = _validated_estimator_config(
        estimator_config,
        # Default: include calibration uncertainty when calibrated
        default_oua_jackknife=calibration_result is not None,
    )
    estimator_obj = CalibratedDirectEstimator(
        target_policies=target_policies,
        reward_calibrator=(
            calibration_result.calibrator if calibration_result else None
        ),
        oua_jackknife=oua_jackknife,
        **estimator_kwargs,
    )

    for policy in target_policies:
        fd = fresh_draws_dict[policy]
        if covariate_names:
            fd = compute_response_covariates(fd, covariate_names=covariate_names)
        estimator_obj.add_fresh_draws(policy, fd)

    results = estimator_obj.fit_and_estimate()

    # --- 7. Inverse-transform results back to the original scale if
    # normalization was applied
    _denormalize_results(results, norm_info, verbose)

    # --- 8. Metadata (mode/mode_selection keys kept for 0.3.0-output compat)
    results.metadata["mode"] = "direct"
    results.metadata["estimator"] = chosen_estimator
    results.metadata["target_policies"] = target_policies
    if fresh_draws_dir:
        results.metadata["fresh_draws_dir"] = fresh_draws_dir
    if fresh_draws_data is not None:
        results.metadata["fresh_draws_source"] = "in_memory"

    # Calibration source metadata
    if calibration_data_path:
        results.metadata["calibration"] = (
            "from_calibration_data_combined"
            if combine_oracle_sources
            else "from_calibration_data_only"
        )
        results.metadata["calibration_data_path"] = calibration_data_path
    else:
        results.metadata["calibration"] = (
            "from_fresh_draws" if calibration_result else "none"
        )
        if calibration_result:
            # Only set oracle_coverage for fresh-draws-only calibration
            results.metadata["oracle_coverage"] = oracle_coverage

    if estimator_config:
        results.metadata["estimator_config"] = dict(estimator_config)

    if oracle_sources_metadata:
        results.metadata["oracle_sources"] = oracle_sources_metadata

    results.metadata["mode_selection"] = {
        "mode": "direct",
        "estimator": chosen_estimator,
        "logprob_coverage": 0.0,  # Direct-only mode has no logged data
        "has_fresh_draws": True,
        "has_logged_data": False,
        "reason": "Direct mode is the only mode in 0.4.x",
    }

    # Add calibrator for transportability audits
    if calibration_result:
        results.calibrator = calibration_result.calibrator

    return results


def _validated_estimator_config(
    estimator_config: Optional[Dict[str, Any]],
    default_oua_jackknife: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate estimator_config and split off the oua_jackknife override.

    A user-supplied oua_jackknife OVERRIDES the pipeline default (instead of
    colliding with it as a duplicate keyword). Remaining keys must be real
    CalibratedDirectEstimator kwargs; unknown keys raise a ValueError listing
    the valid ones. reward_calibrator is managed by the pipeline and rejected.

    Returns:
        (oua_jackknife, remaining kwargs to forward to the estimator)
    """
    cfg = dict(estimator_config or {})

    if "reward_calibrator" in cfg:
        raise ValueError(
            "reward_calibrator is managed by analyze_dataset (it is fit from "
            "your oracle labels); remove it from estimator_config. To supply "
            "your own calibrator, construct CalibratedDirectEstimator "
            "directly (see cje.advanced)."
        )

    raw_oua = cfg.pop("oua_jackknife", default_oua_jackknife)
    oua_jackknife = bool(raw_oua)

    unknown = sorted(key for key in cfg if key not in _ESTIMATOR_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown estimator_config key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(_ESTIMATOR_CONFIG_KEYS)}."
        )

    return oua_jackknife, cfg


def _build_covariate_list(
    calibration_covariates: Optional[List[str]],
    include_response_length: bool,
    dataset: Optional[Dataset] = None,
) -> Optional[List[str]]:
    """
    Build the final covariate list, prepending auto-computable covariates if needed.

    Args:
        calibration_covariates: User-specified covariate names (or None)
        include_response_length: Whether to prepend the response_length covariate
        dataset: Optional dataset to validate against (checks for response field)

    Returns:
        List of covariate names to use, or None if no covariates specified
    """
    # Start with user-specified covariates
    covariates = list(calibration_covariates or [])

    # Prepend response_length if flag is set
    if include_response_length:
        # Validate that dataset has response field
        if dataset is not None:
            for i, sample in enumerate(dataset.samples):
                if not hasattr(sample, "response") or sample.response is None:
                    raise ValueError(
                        f"include_response_length=True requires all samples to have a 'response' field. "
                        f"Sample {i} (prompt_id={sample.prompt_id}) is missing this field or it is None."
                    )

        # Add response_length to the front of the list if not already there
        if "response_length" not in covariates:
            covariates.insert(0, "response_length")

    return covariates if covariates else None


def _denormalize_results(
    results: EstimationResult,
    norm_info: Optional[Any],
    verbose: bool,
) -> None:
    """Inverse-transform results back to the original score scale in place.

    The two normalization sources share one inverse transform; only the
    ScaleInfo consulted differs: the oracle-label scale when oracle labels
    exist, otherwise the judge-score scale (tagged results_scale =
    "judge_original" in the normalization metadata). No-op when no
    normalization was applied.
    """
    if norm_info is None:
        return

    if norm_info.oracle_label_scale:
        scale = norm_info.oracle_label_scale
        results_scale = None
        scale_label = "oracle"
    else:
        # No oracle labels, use judge scale for inverse transform
        scale = norm_info.judge_score_scale
        results_scale = "judge_original"
        scale_label = "judge"

    # Transform estimates and standard errors back to original scale
    results.estimates = scale.inverse_array(results.estimates)
    # Standard errors scale linearly with the range
    scale_factor = scale.max_val - scale.min_val
    results.standard_errors = results.standard_errors * scale_factor

    # Paired bootstrap replicate matrix: replicates are estimates, so they
    # get the same affine inverse transform (compare_policies then returns
    # differences/SEs/CIs on the original scale)
    if results.bootstrap_samples is not None:
        results.bootstrap_samples = (
            results.bootstrap_samples * scale_factor + scale.min_val
        )

    # Analytic pairwise inference: SEs scale linearly, variances
    # quadratically (differences are shift-invariant — no min_val offset)
    pairwise = results.metadata.get("pairwise_inference")
    if isinstance(pairwise, dict):
        for pair_entry in pairwise.values():
            if not isinstance(pair_entry, dict):
                continue
            for se_key in ("se", "se_sampling"):
                se_value = pair_entry.get(se_key)
                if isinstance(se_value, (int, float)):
                    pair_entry[se_key] = float(se_value * scale_factor)
            var_value = pair_entry.get("var_oua_diff")
            if isinstance(var_value, (int, float)):
                pair_entry["var_oua_diff"] = float(var_value * scale_factor**2)

    # Also denormalize bootstrap CIs if present (metadata mirror AND the
    # typed ci_info record must stay in sync)
    if "bootstrap_ci" in results.metadata:
        boot_ci = results.metadata["bootstrap_ci"]
        boot_ci["lower"] = [
            float(v * scale_factor + scale.min_val) for v in boot_ci["lower"]
        ]
        boot_ci["upper"] = [
            float(v * scale_factor + scale.min_val) for v in boot_ci["upper"]
        ]
        if results.ci_info is not None and results.ci_info.method == "percentile":
            results.ci_info.lower = list(boot_ci["lower"])
            results.ci_info.upper = list(boot_ci["upper"])

    # Add normalization metadata
    results.metadata["normalization"] = norm_info.to_dict()
    if results_scale is not None:
        results.metadata["normalization"]["results_scale"] = results_scale

    if verbose:
        suffix = " (no oracle labels)" if results_scale is not None else ""
        logger.info(
            f"Inverse-transformed results to original {scale_label} scale "
            f"[{scale.min_val}, {scale.max_val}]{suffix}"
        )


def _combine_oracle_sources(
    calibration_dataset: Optional[Dataset],
    logged_dataset: Optional[Dataset],
    fresh_draws_per_policy: Optional[Dict[str, Any]],
    target_policies: List[str],
    judge_field: str,
    oracle_field: str,
    verbose: bool = False,
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Combine oracle (judge, oracle) pairs from multiple sources.

    Oracle labels attach to RESPONSES, not prompts: several sources (or
    several policies' fresh draws) can legitimately contribute different
    pairs for the same prompt_id, and ALL of them are kept as calibration
    pairs. Only true duplicates — same response identity (source; for
    fresh draws also policy and draw index), same prompt, identical judge
    and oracle values — are deduped. Cross-source disagreements on the
    same prompt (oracle diff > 0.05) are reported as conflicts but both
    pairs still enter calibration.

    Args:
        calibration_dataset: Optional calibration dataset with oracle labels
        logged_dataset: Optional logged dataset (can be None in Direct mode)
        fresh_draws_per_policy: Optional dict of fresh draws by policy
        target_policies: List of target policy names (for dataset construction)
        judge_field: Field name for judge scores
        oracle_field: Field name for oracle labels
        verbose: Whether to log progress

    Returns:
        Tuple of (combined_dataset, oracle_sources_metadata)
    """
    from ..data.models import Sample

    # Accumulate every (prompt_id, source, judge, oracle) pair
    pairs: List[Tuple[str, str, float, float]] = []
    seen_pairs: set = set()
    oracle_by_prompt: Dict[str, List[Tuple[str, float]]] = {}
    conflicts: List[Dict[str, Any]] = []

    def _add_pair(
        prompt_id: str, source: str, judge_val: float, oracle_val: float
    ) -> bool:
        """Add a calibration pair; returns False for true duplicates.

        The source string identifies the response, not just the source
        family: fresh-draw pairs arrive as "fresh_draws:<policy>:draw<i>",
        so distinct policies' (or draws') labels for one prompt never
        collapse even when their (judge, oracle) values coincide — which
        is common with binary or rubric scores. Conflicts are still
        detected at the source-FAMILY level (calibration_data vs
        fresh_draws vs logged_data): different responses within one
        family legitimately earn different labels.
        """
        key = (source, prompt_id, judge_val, oracle_val)
        if key in seen_pairs:
            return False
        # Cross-source-family conflict check (pairs are kept either way)
        source_family = source.split(":", 1)[0]
        for prev_source, prev_oracle in oracle_by_prompt.get(prompt_id, []):
            prev_family = prev_source.split(":", 1)[0]
            if prev_family != source_family and abs(prev_oracle - oracle_val) > 0.05:
                conflicts.append(
                    {
                        "prompt_id": prompt_id,
                        "existing_value": float(prev_oracle),
                        "existing_source": prev_source,
                        "new_value": float(oracle_val),
                        "new_source": source,
                        "difference": abs(prev_oracle - oracle_val),
                    }
                )
                break
        seen_pairs.add(key)
        pairs.append((prompt_id, source, judge_val, oracle_val))
        oracle_by_prompt.setdefault(prompt_id, []).append((source, oracle_val))
        return True

    # Logged data
    n_from_logged = 0
    if logged_dataset:
        for sample in logged_dataset.samples:
            oracle_val = (
                sample.oracle_label
                if oracle_field == "oracle_label"
                else sample.metadata.get(oracle_field)
            )
            if oracle_val is not None:
                judge_val = (
                    sample.judge_score
                    if judge_field == "judge_score"
                    else sample.metadata.get(judge_field)
                )
                if judge_val is not None:
                    if _add_pair(
                        sample.prompt_id,
                        "logged_data",
                        float(judge_val),
                        float(oracle_val),
                    ):
                        n_from_logged += 1

    # Fresh draws (every policy's labeled draws count — labels attach to
    # responses, so K policies can contribute K pairs for one prompt)
    n_from_fresh = 0
    if fresh_draws_per_policy:
        for policy, fd_dataset in fresh_draws_per_policy.items():
            for fd_sample in fd_dataset.samples:
                if (
                    fd_sample.oracle_label is not None
                    and fd_sample.judge_score is not None
                ):
                    # Identify the response (policy + draw), not just the
                    # family: a bare "fresh_draws" source collapsed
                    # distinct policies' draws for one prompt whenever
                    # their (judge, oracle) values coincided.
                    draw_idx = getattr(fd_sample, "draw_idx", 0)
                    if _add_pair(
                        fd_sample.prompt_id,
                        f"fresh_draws:{policy}:draw{draw_idx}",
                        float(fd_sample.judge_score),
                        float(fd_sample.oracle_label),
                    ):
                        n_from_fresh += 1

    # Calibration data
    n_from_calib = 0
    if calibration_dataset:
        for sample in calibration_dataset.samples:
            oracle_val = (
                sample.oracle_label
                if oracle_field == "oracle_label"
                else sample.metadata.get(oracle_field)
            )
            if oracle_val is not None:
                judge_val = (
                    sample.judge_score
                    if judge_field == "judge_score"
                    else sample.metadata.get(judge_field)
                )
                if judge_val is not None:
                    if _add_pair(
                        sample.prompt_id,
                        "calibration_data",
                        float(judge_val),
                        float(oracle_val),
                    ):
                        n_from_calib += 1

    # Log conflicts if any (loud by default — cross-source disagreement is
    # a data-quality signal the user must see)
    if conflicts:
        logger.warning(
            f"Found {len(conflicts)} prompts with conflicting oracle labels "
            f"across sources (diff > 0.05). All pairs are kept for "
            f"calibration (oracle labels attach to responses, not prompts); "
            f"large cross-source differences may indicate inconsistent "
            f"oracle definitions."
        )
        # Log top 5 conflicts
        for conflict in conflicts[:5]:
            logger.debug(
                f"  {conflict['prompt_id']}: {conflict['existing_source']}={conflict['existing_value']:.3f} "
                f"vs {conflict['new_source']}={conflict['new_value']:.3f} (diff={conflict['difference']:.3f})"
            )

    # Build combined dataset with every accumulated pair
    combined_samples = []
    for prompt_id, source, judge_val, oracle_val in pairs:
        # Create Sample with judge and oracle
        sample = Sample(
            prompt_id=prompt_id,
            prompt="",  # Not needed for calibration
            response="",
            reward=None,  # Will be calibrated
            judge_score=judge_val,
            oracle_label=oracle_val,
            metadata={"source": source},
        )
        combined_samples.append(sample)

    if verbose:
        logger.info(
            f"Combined oracle sources: {len(combined_samples)} total pairs "
            f"(calib={n_from_calib}, fresh={n_from_fresh}, logged={n_from_logged})"
        )

    # Build metadata (counts reflect pairs that actually enter calibration)
    oracle_sources_metadata: Dict[str, Any] = {
        "calibration_data": {
            "n_oracle": n_from_calib,
            "coverage": (
                n_from_calib / calibration_dataset.n_samples
                if calibration_dataset
                else 0.0
            ),
        },
        "logged_data": {
            "n_oracle": n_from_logged,
            "coverage": (
                n_from_logged / logged_dataset.n_samples if logged_dataset else 0.0
            ),
        },
        "fresh_draws": {
            "n_oracle": n_from_fresh,
            "coverage": None,  # Can't compute without knowing total fresh draws
        },
        "total_oracle": len(combined_samples),
        "n_conflicts": len(conflicts),
    }

    if conflicts:
        # Limit to top 10 for metadata size
        oracle_sources_metadata["conflicts"] = conflicts[:10]

    # Create combined dataset using target_policies parameter
    combined_dataset = Dataset(
        samples=combined_samples,
        target_policies=target_policies,
        metadata={"combined_oracle_sources": True},
    )

    return combined_dataset, oracle_sources_metadata

"""
High-level analysis functions for CJE.

This module provides simple, one-line analysis functions that handle
the complete CJE workflow automatically. The whole Direct-mode pipeline
lives here: validate inputs -> load draws -> resolve calibration source ->
calibrate -> CalibratedDirectEstimator -> estimate -> denormalize ->
metadata.
"""

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..calibration import calibrate_dataset
from ..data.models import Dataset, EstimationResult
from ..data.normalization import ScaleDeclaration, validate_values_on_scale
from ._removed import validate_estimator_name

if TYPE_CHECKING:
    from ..diagnostics.transport import TransportAuditConfig

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
    *,
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
    fresh_judge_scale: ScaleDeclaration = None,
    fresh_oracle_scale: ScaleDeclaration = None,
    calibration_judge_scale: ScaleDeclaration = None,
    calibration_oracle_scale: ScaleDeclaration = None,
    output_scale: ScaleDeclaration = None,
    strict: bool = False,
    on_invalid: Optional[str] = None,
    label_design: str = "representative",
    label_propensities: Optional[Dict[str, Any]] = None,
    transport: Optional["TransportAuditConfig"] = None,
) -> EstimationResult:
    """
    Analyze policies using fresh draws (Direct mode).

    This high-level function handles:
    - Data loading and validation
    - Automatic reward calibration (judge → oracle mapping)
    - Oracle source combining (pooling labels from multiple sources)
    - Complete analysis workflow

    Args:
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
        fresh_judge_scale: Optional declared ``(minimum, maximum)`` scale for
            evaluation judge scores. Without it, out-of-unit fresh scores keep
            the legacy observed-range normalization behavior.
        fresh_oracle_scale: Optional declared scale for oracle labels embedded
            in the evaluation draws.
        calibration_judge_scale: Optional declared scale for judge scores in
            ``calibration_data_path``. External calibration data defaults to
            the unit interval and does not infer its scale from observations.
        calibration_oracle_scale: Optional declared scale for oracle labels in
            ``calibration_data_path``.
        output_scale: Optional result ``(minimum, maximum)`` display scale. By
            default, calibrated results use their oracle source scale and
            uncalibrated results use the evaluation judge scale. Declaring a
            scale changes the display axis only — never the estimand label —
            and is ignored (with a warning) for mixed direct-oracle/raw-judge
            runs, which stay on the internal unit scale.
        strict: Retained for compatibility; invalid records already raise by
            default. Pass ``on_invalid="drop"`` to drop them instead.
        on_invalid: Invalid-record policy, ``"error"`` (default) or
            ``"drop"``. Dropping records per-policy counts in
            ``metadata["n_invalid_dropped_per_policy"]`` (plus the
            ``n_invalid_dropped`` total) and warns once with the counts.
        label_design: Oracle-label sampling design: ``"representative"``,
            ``"known_propensity"``, or ``"targeted_unknown"``.
        label_propensities: Per-policy inclusion-probability vectors required
            when ``label_design="known_propensity"``.
        transport: Optional held-out residual-audit configuration. Probe rows
            use the same public judge/oracle units and field names as this
            call. Policies without probes are recorded as ``NOT_CHECKED``;
            probes without a declared per-policy margin are ``NOT_GRADED``.

    Returns:
        EstimationResult with estimates, standard errors, and metadata.

        New metadata fields when using calibration_data_path:
        - results.metadata["oracle_sources"]: Breakdown of oracle labels by source
        - results.metadata["oracle_sources"]["conflicts"]: Cross-source oracle disagreements
          on rows sharing an explicit observation_id
        - results.metadata["oracle_sources"]["value_conflicts"]: Cross-source oracle
          disagreements on id-less rows sharing (prompt_id, judge_score)

    Raises:
        ValueError: If required data is missing or declarations are invalid.

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
    from ..data.loaders import DatasetLoader
    from ..data.models import Sample
    from ..data.normalization import (
        ScaledCalibrator,
        coerce_scale,
        unit_scale,
    )
    from ..diagnostics.robust_inference import CalibrationProvenance, LabelDesign
    from ..estimators.direct_method import CalibratedDirectEstimator

    # --- 1. Presence check: at least one fresh-draws source
    if fresh_draws_dir is None and fresh_draws_data is None:
        raise ValueError(
            "Must provide at least one of: fresh_draws_dir, fresh_draws_data"
        )
    if fresh_draws_dir is not None and fresh_draws_data is not None:
        raise ValueError(
            "Provide exactly one evaluation source: fresh_draws_dir or "
            "fresh_draws_data, not both."
        )
    # Loud by default: invalid records raise unless the caller explicitly
    # opts into dropping (strict is a compatibility no-op — the default
    # already errors).
    invalid_mode = on_invalid or "error"
    if invalid_mode not in {"error", "drop"}:
        raise ValueError("on_invalid must be 'error' or 'drop'")
    # Per-policy counts of dropped fresh-draw records, accumulated across
    # the file/directory readers and fresh_draws_from_dict when dropping.
    fresh_drop_stats: Dict[str, int] = {}

    # --- 2. Estimator-name validation ("auto" resolves to "direct"; removed
    # 0.3.x OPE names raise the migration error)
    chosen_estimator = (estimator or "auto").strip().lower()
    if chosen_estimator == "auto":
        chosen_estimator = "direct"
    else:
        validate_estimator_name(chosen_estimator)

    # --- 3. Load fresh draws (in-memory dict, directory, or single
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
            raw_fresh_draws = fresh_draws_data_from_file(
                draws_path, on_invalid=invalid_mode, drop_stats=fresh_drop_stats
            )
        else:
            raw_fresh_draws = fresh_draws_data_from_dir(
                draws_path,
                verbose=verbose,
                judge_field=judge_field,
                oracle_field=oracle_field,
                on_invalid=invalid_mode,
                exclude_paths=(
                    [Path(calibration_data_path)] if calibration_data_path else None
                ),
                drop_stats=fresh_drop_stats,
            )

    fresh_draws_dict, norm_info = fresh_draws_from_dict(
        raw_fresh_draws,
        verbose=verbose,
        auto_normalize=True,
        judge_field=judge_field,
        oracle_field=oracle_field,
        judge_scale=fresh_judge_scale,
        oracle_scale=fresh_oracle_scale,
        on_invalid=invalid_mode,
        drop_stats=fresh_drop_stats,
    )
    fresh_judge_public_scale = (
        norm_info.judge_score_scale
        if norm_info is not None
        else coerce_scale(fresh_judge_scale, field_name="fresh_judge_scale")
        or unit_scale()
    )
    fresh_oracle_public_scale = (
        norm_info.oracle_label_scale
        if norm_info is not None and norm_info.oracle_label_scale is not None
        else coerce_scale(fresh_oracle_scale, field_name="fresh_oracle_scale")
        or unit_scale()
    )
    target_policies = sorted(fresh_draws_dict.keys())

    if verbose:
        logger.info(
            f"Found {len(target_policies)} policies: {', '.join(target_policies)}"
        )

    all_fresh_draws = []
    for fd in fresh_draws_dict.values():
        all_fresh_draws.extend(fd.samples)
    full_oracle_policies = {
        policy
        for policy, dataset in fresh_draws_dict.items()
        if dataset.samples
        and all(sample.oracle_label is not None for sample in dataset.samples)
    }
    full_evaluation_oracle_coverage = bool(target_policies) and len(
        full_oracle_policies
    ) == len(target_policies)

    # --- 5. Resolve the calibration source and calibrate:
    # calibration_data_path (optionally pooled with labeled draws via
    # _combine_oracle_sources) OR oracle-in-draws
    oracle_sources_metadata = None
    calibration_result = None
    calibration_dataset_for_rewards: Optional[Dataset] = None
    calibration_provenance = None
    covariate_names = None
    oracle_coverage = 0.0
    n_calibration_clusters = 0
    calibration_oracle_public_scale = unit_scale()
    calibration_judge_public_scale = unit_scale()

    if calibration_data_path:
        if verbose:
            logger.info(f"Loading calibration dataset from {calibration_data_path}")
        calibration_loader = DatasetLoader(
            judge_field=judge_field,
            oracle_field=oracle_field,
            judge_scale=calibration_judge_scale,
            oracle_scale=calibration_oracle_scale,
            strict=strict,
            on_invalid=invalid_mode,
        )
        calibration_dataset = calibration_loader.load_from_jsonl(calibration_data_path)
        calibration_judge_public_scale = calibration_loader.judge_scale
        calibration_oracle_public_scale = calibration_loader.oracle_scale
        for sample in calibration_dataset.samples:
            sample.metadata["row_role"] = "external"
            sample.metadata["source"] = "calibration_data"
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
                    "judge_score",
                    "oracle_label",
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

        n_fit_labels = (
            sum(
                sample.oracle_label is not None
                for sample in calibration_dataset_for_rewards.samples
            )
            if calibration_dataset_for_rewards is not None
            else 0
        )
        n_calibration_clusters = (
            len(
                {
                    str(sample.prompt_id)
                    for sample in calibration_dataset_for_rewards.samples
                    if sample.oracle_label is not None
                }
            )
            if calibration_dataset_for_rewards is not None
            else 0
        )
        if n_calibration_clusters >= 4:
            assert calibration_dataset_for_rewards is not None
            covariate_names = _build_covariate_list(
                calibration_covariates,
                include_response_length,
                calibration_dataset_for_rewards,
            )
            _, calibration_result = calibrate_dataset(
                calibration_dataset_for_rewards,
                judge_field="judge_score",
                oracle_field="oracle_label",
                enable_cross_fit=True,
                covariate_names=covariate_names,
            )
            calibration_provenance = _build_calibration_provenance(
                calibration_dataset_for_rewards,
                covariate_names,
                CalibrationProvenance,
            )
        else:
            covariate_names = _build_covariate_list(
                calibration_covariates,
                include_response_length,
                calibration_dataset_for_rewards or calibration_dataset,
            )
            if full_evaluation_oracle_coverage:
                logger.warning(
                    "Only %d independent oracle-labeled prompt clusters are "
                    "available (<4; %d labeled rows); a calibrator cannot be "
                    "fit, so using the complete evaluation oracle labels "
                    "directly.",
                    n_calibration_clusters,
                    n_fit_labels,
                )
            else:
                logger.warning(
                    "Only %d independent oracle-labeled prompt clusters are "
                    "available (<4; %d labeled rows); returning the UNCALIBRATED "
                    "raw-judge tier instead of failing the entire run.",
                    n_calibration_clusters,
                    n_fit_labels,
                )
    else:
        # Oracle-in-draws: learn calibration from labeled fresh draws
        n_with_oracle = sum(1 for s in all_fresh_draws if s.oracle_label is not None)
        n_calibration_clusters = len(
            {
                str(sample.prompt_id)
                for sample in all_fresh_draws
                if sample.oracle_label is not None
            }
        )
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
                metadata = dict(fd_sample.metadata)
                metadata.update(
                    {
                        "source": "fresh_draws",
                        "row_role": "evaluation",
                        "evaluation_key": (
                            fd_sample.target_policy,
                            fd_sample.prompt_id,
                            fd_sample.draw_idx,
                        ),
                    }
                )
                sample = Sample(
                    prompt_id=fd_sample.prompt_id,
                    prompt="",  # Not needed for calibration
                    response=fd_sample.response or "",
                    reward=None,  # Will be calibrated
                    judge_score=fd_sample.judge_score,
                    oracle_label=fd_sample.oracle_label,
                    metadata=metadata,
                    row_id=fd_sample.row_id,
                    observation_id=fd_sample.observation_id,
                    source_id=fd_sample.source_id,
                )
                calibration_samples.append(sample)
            fresh_dataset = Dataset(
                samples=calibration_samples, target_policies=target_policies
            )

            calibration_dataset_for_rewards = fresh_dataset
            if n_calibration_clusters >= 4:
                covariate_names = _build_covariate_list(
                    calibration_covariates, include_response_length, fresh_dataset
                )
                _, calibration_result = calibrate_dataset(
                    fresh_dataset,
                    judge_field="judge_score",
                    oracle_field="oracle_label",
                    enable_cross_fit=True,
                    covariate_names=covariate_names,
                )
                calibration_provenance = _build_calibration_provenance(
                    fresh_dataset,
                    covariate_names,
                    CalibrationProvenance,
                )
            else:
                covariate_names = _build_covariate_list(
                    calibration_covariates, include_response_length, fresh_dataset
                )
                if full_evaluation_oracle_coverage:
                    logger.warning(
                        "Only %d independent oracle-labeled prompt clusters are "
                        "available (<4; %d labeled rows); a calibrator cannot be "
                        "fit, so using the complete evaluation oracle labels "
                        "directly.",
                        n_calibration_clusters,
                        n_with_oracle,
                    )
                else:
                    logger.warning(
                        "Only %d independent oracle-labeled prompt clusters are "
                        "available (<4; %d labeled rows); returning the "
                        "UNCALIBRATED raw-judge tier instead of failing the entire run.",
                        n_calibration_clusters,
                        n_with_oracle,
                    )
        else:
            requested_covariates = _build_covariate_list(
                calibration_covariates, include_response_length
            )
            if requested_covariates:
                for fresh_draw_dataset in fresh_draws_dict.values():
                    compute_response_covariates(
                        fresh_draw_dataset, covariate_names=requested_covariates
                    )
            logger.warning(
                "No oracle labels found — returning UNCALIBRATED judge-score "
                "means; CIs do not account for judge bias. Results will be "
                "labeled method='naive_direct'. Add oracle labels to the "
                "fresh draws or provide calibration_data_path to calibrate."
            )

    direct_oracle_without_calibration_policies = (
        set(full_oracle_policies) if calibration_result is None else set()
    )
    direct_oracle_without_calibration = bool(target_policies) and len(
        direct_oracle_without_calibration_policies
    ) == len(target_policies)
    uncalibrated_policies = (
        [
            policy
            for policy in target_policies
            if policy not in direct_oracle_without_calibration_policies
        ]
        if calibration_result is None
        else []
    )
    mixed_direct_without_calibration = bool(
        direct_oracle_without_calibration_policies and uncalibrated_policies
    )

    # --- 6. Build the estimator (validating estimator_config), attach draws
    # (computing covariates when the calibrator expects them), and estimate
    oua_jackknife, estimator_kwargs = _validated_estimator_config(
        estimator_config,
        # Default: include calibration uncertainty when calibrated
        default_oua_jackknife=calibration_result is not None,
    )
    if direct_oracle_without_calibration and "inference_method" not in (
        estimator_config or {}
    ):
        # Full-oracle analysis needs no calibrator refit. Keep the high-level
        # default lightweight while honoring an explicitly requested bootstrap.
        estimator_kwargs["inference_method"] = "cluster_robust"
    label_design_obj = _build_label_design(
        label_design,
        label_propensities,
        fresh_draws_dict,
        LabelDesign,
    )
    estimator_obj = CalibratedDirectEstimator(
        target_policies=target_policies,
        reward_calibrator=(
            calibration_result.calibrator if calibration_result else None
        ),
        oua_jackknife=oua_jackknife,
        calibration_provenance=calibration_provenance,
        label_design=label_design_obj,
        **estimator_kwargs,
    )

    for policy in target_policies:
        fd = fresh_draws_dict[policy]
        if calibration_result is not None and covariate_names:
            fd = compute_response_covariates(fd, covariate_names=covariate_names)
        if calibration_result is None and policy in uncalibrated_policies:
            # The graded fallback is specifically the raw-judge estimand.
            # Keeping partial/full oracle labels here would make the core route
            # to augmentation or a direct oracle mean despite having no fitted
            # calibrator, while the result was labeled RAW_JUDGE_MEAN.
            fd = fd.model_copy(
                update={
                    "samples": [
                        sample.model_copy(update={"oracle_label": None})
                        for sample in fd.samples
                    ]
                }
            )
        estimator_obj.add_fresh_draws(policy, fd)

    results = estimator_obj.fit_and_estimate()
    declared_output_scale = coerce_scale(output_scale, field_name="output_scale")
    if mixed_direct_without_calibration:
        # Oracle and judge values can have different public scales. A single
        # result vector cannot honestly claim either one — a declared
        # output_scale must not project the heterogeneous internals onto one
        # axis either — so retain unit-scale values and expose the per-policy
        # estimands below.
        if declared_output_scale is not None:
            logger.warning(
                "Ignoring output_scale=(%s, %s): mixed direct-oracle/raw-judge "
                "results stay on the internal unit scale because a single "
                "axis cannot honestly represent both estimands.",
                declared_output_scale.min_val,
                declared_output_scale.max_val,
            )
        result_output_scale = unit_scale()
        result_scale_source = "mixed_direct_oracle_raw_judge"
    elif declared_output_scale is not None:
        result_output_scale = declared_output_scale
        result_scale_source = "declared"
    elif (
        calibration_data_path
        and calibration_result is not None
        and oracle_sources_metadata is not None
        and oracle_sources_metadata.get("calibration_data", {}).get("n_oracle", 0) > 0
    ):
        result_output_scale = calibration_oracle_public_scale
        result_scale_source = "calibration_oracle"
    elif calibration_result is not None:
        result_output_scale = fresh_oracle_public_scale
        result_scale_source = "fresh_oracle"
    elif direct_oracle_without_calibration:
        result_output_scale = fresh_oracle_public_scale
        result_scale_source = "fresh_oracle_direct"
    else:
        result_output_scale = fresh_judge_public_scale
        result_scale_source = "fresh_judge_uncalibrated"

    # The estimand label derives from the calibration/claim-tier state alone;
    # a declared output_scale changes the display axis, never the estimand.
    if calibration_result is not None or direct_oracle_without_calibration:
        result_estimand = "oracle_mean"
    elif mixed_direct_without_calibration:
        result_estimand = "mixed"
    else:
        result_estimand = "judge_mean"

    # --- 7. Transform every public result artifact exactly once from the
    # internal unit scale into one declared output scale.
    _denormalize_results(
        results,
        output_scale=result_output_scale,
        judge_input_scale=fresh_judge_public_scale,
        fresh_norm_info=norm_info,
        result_scale_source=result_scale_source,
        result_estimand=result_estimand,
        verbose=verbose,
    )

    # --- 8. Metadata (mode/mode_selection keys kept for 0.3.0-output compat)
    results.metadata["mode"] = "direct"
    results.metadata["estimator"] = chosen_estimator
    results.metadata["target_policies"] = target_policies
    if fresh_draws_dir:
        results.metadata["fresh_draws_dir"] = fresh_draws_dir
    if fresh_draws_data is not None:
        results.metadata["fresh_draws_source"] = "in_memory"

    # Dropping is an explicit opt-in: record what was dropped per policy and
    # warn once with the counts so a partially corrupt evaluation file can
    # never silently shift estimates.
    if invalid_mode == "drop":
        n_invalid_dropped = int(sum(fresh_drop_stats.values()))
        results.metadata["n_invalid_dropped"] = n_invalid_dropped
        results.metadata["n_invalid_dropped_per_policy"] = {
            policy: int(count) for policy, count in sorted(fresh_drop_stats.items())
        }
        if n_invalid_dropped:
            logger.warning(
                "on_invalid='drop': dropped %d invalid fresh-draw record(s) "
                "(%s). Estimates use the remaining rows only; pass "
                "on_invalid='error' to fail on invalid records instead.",
                n_invalid_dropped,
                ", ".join(
                    f"{policy}: {count}"
                    for policy, count in sorted(fresh_drop_stats.items())
                ),
            )

    # Calibration source metadata
    if calibration_data_path:
        results.metadata["calibration"] = (
            (
                "from_calibration_data_combined"
                if combine_oracle_sources
                else "from_calibration_data_only"
            )
            if calibration_result is not None
            else (
                "direct_oracle"
                if direct_oracle_without_calibration
                else (
                    "mixed_direct"
                    if mixed_direct_without_calibration
                    else "insufficient_labels"
                )
            )
        )
        results.metadata["calibration_data_path"] = calibration_data_path
    else:
        results.metadata["calibration"] = (
            "from_fresh_draws"
            if calibration_result
            else (
                "direct_oracle"
                if direct_oracle_without_calibration
                else "mixed_direct" if mixed_direct_without_calibration else "none"
            )
        )
        if calibration_result:
            # Only set oracle_coverage for fresh-draws-only calibration
            results.metadata["oracle_coverage"] = oracle_coverage

    if estimator_config:
        results.metadata["estimator_config"] = dict(estimator_config)

    if oracle_sources_metadata:
        _scale_oracle_source_metadata(oracle_sources_metadata, result_output_scale)
        results.metadata["oracle_sources"] = oracle_sources_metadata

    point_routes = list(results.metadata.get("point_estimator", {}).get("routes", []))
    route_by_policy = {
        policy: (point_routes[index] if index < len(point_routes) else "unknown")
        for index, policy in enumerate(target_policies)
    }
    calibration_status_by_policy = {
        policy: (
            "DIRECT_ORACLE"
            if route_by_policy[policy] == "direct_oracle"
            else "CALIBRATED" if calibration_result is not None else "UNCALIBRATED"
        )
        for policy in target_policies
    }
    claim_tier_by_policy = {
        policy: (
            "DIRECT_ORACLE_MEAN"
            if route_by_policy[policy] == "direct_oracle"
            else (
                "CALIBRATED_ORACLE_MEAN"
                if calibration_result is not None
                else "RAW_JUDGE_MEAN"
            )
        )
        for policy in target_policies
    }
    results.metadata["calibration_status_by_policy"] = calibration_status_by_policy
    results.metadata["claim_tier_by_policy"] = claim_tier_by_policy
    results.metadata["calibration_status"] = (
        "CALIBRATED"
        if calibration_result is not None
        else (
            "DIRECT_ORACLE"
            if direct_oracle_without_calibration
            else "MIXED" if mixed_direct_without_calibration else "UNCALIBRATED"
        )
    )
    results.metadata["claim_tier"] = (
        "CALIBRATED_ORACLE_MEAN"
        if calibration_result is not None
        else (
            "DIRECT_ORACLE_MEAN"
            if direct_oracle_without_calibration
            else "MIXED" if mixed_direct_without_calibration else "RAW_JUDGE_MEAN"
        )
    )
    results.metadata["data_provenance"] = _analysis_provenance_metadata(
        fresh_draws_dict=fresh_draws_dict,
        calibration_dataset=calibration_dataset_for_rewards,
        fresh_judge_scale=fresh_judge_public_scale,
        fresh_oracle_scale=fresh_oracle_public_scale,
        calibration_judge_scale=(
            calibration_judge_public_scale if calibration_data_path else None
        ),
        calibration_oracle_scale=(
            calibration_oracle_public_scale if calibration_data_path else None
        ),
        output_scale=result_output_scale,
        invalid_mode=invalid_mode,
        covariate_names=covariate_names,
    )

    results.metadata["mode_selection"] = {
        "mode": "direct",
        "estimator": chosen_estimator,
        "logprob_coverage": 0.0,  # Direct-only mode has no logged data
        "has_fresh_draws": True,
        "has_logged_data": False,
        "reason": "Direct mode is the only supported estimator family",
    }

    # Add calibrator for transportability audits
    if calibration_result:
        results.calibrator = ScaledCalibrator(
            calibration_result.calibrator,
            judge_scale=fresh_judge_public_scale,
            output_scale=result_output_scale,
        )
    elif uncalibrated_policies:
        _mark_uncalibrated_result(
            results,
            uncalibrated_policies,
            n_label_clusters=n_calibration_clusters,
        )

    _attach_transport_audits(
        results,
        config=transport,
        target_policies=target_policies,
        calibration_dataset=calibration_dataset_for_rewards,
        judge_field=judge_field,
        oracle_field=oracle_field,
        oracle_input_scale=fresh_oracle_public_scale,
        output_scale=result_output_scale,
    )

    return results


def _attach_transport_audits(
    results: EstimationResult,
    *,
    config: Optional[Any],
    target_policies: List[str],
    calibration_dataset: Optional[Dataset],
    judge_field: str,
    oracle_field: str,
    oracle_input_scale: Any,
    output_scale: Any,
) -> None:
    """Run configured held-out probes and record every policy's audit state."""
    from ..data.ingest import canonicalize_record
    from ..diagnostics import Status
    from ..diagnostics.transport import TransportAuditConfig, audit_transportability

    if config is not None and not isinstance(config, TransportAuditConfig):
        raise TypeError("transport must be a TransportAuditConfig")

    configured_policies = set(config.probes_by_policy) if config is not None else set()
    margin_policies = set(config.delta_max_by_policy) if config is not None else set()
    unknown = sorted((configured_policies | margin_policies) - set(target_policies))
    if unknown:
        raise ValueError(
            "transport configuration contains unknown policies: " + ", ".join(unknown)
        )

    used_row_ids = set()
    used_observation_ids = set()
    if calibration_dataset is not None:
        for sample in calibration_dataset.samples:
            if sample.oracle_label is None:
                continue
            if sample.source_id is not None and sample.row_id is not None:
                used_row_ids.add((str(sample.source_id), str(sample.row_id)))
            if sample.observation_id is not None:
                used_observation_ids.add(str(sample.observation_id))

    def _not_checked(policy: str, reason_code: str, action: str) -> Dict[str, Any]:
        margin = config.delta_max_by_policy.get(policy) if config is not None else None
        return {
            "status": "NOT_CHECKED",
            "performed": False,
            "graded": False,
            "reason_code": reason_code,
            "recommended_action": action,
            "group_label": f"policy:{policy}",
            "delta_max": margin,
            "n_probe": 0,
        }

    audits: Dict[str, Dict[str, Any]] = {}
    point_routes = list(results.metadata.get("point_estimator", {}).get("routes", []))
    route_by_policy = {
        policy: (point_routes[index] if index < len(point_routes) else "unknown")
        for index, policy in enumerate(target_policies)
    }
    for policy in target_policies:
        raw_probes = (
            list(config.probes_by_policy.get(policy, ())) if config is not None else []
        )
        if config is not None and policy in config.probes_by_policy and not raw_probes:
            raise ValueError(
                f"Transport probe collection for policy {policy!r} is empty. "
                "Omit the policy to record NOT_CHECKED."
            )
        if not raw_probes:
            audits[policy] = _not_checked(
                policy,
                "probe_not_provided",
                "supply independent oracle probes to grade residual transport",
            )
            continue
        if results.calibrator is None:
            audits[policy] = _not_checked(
                policy,
                "calibrator_unavailable",
                "fit a calibrated result before auditing residual transport",
            )
            continue

        covariate_names = list(
            getattr(results.calibrator, "covariate_names", None) or []
        )
        probes: List[Any] = []
        for index, raw_probe in enumerate(raw_probes):
            if isinstance(raw_probe, dict):
                canonical = canonicalize_record(
                    raw_probe,
                    index,
                    source_id=f"transport_probe:{policy}",
                    policy=policy,
                    judge_field=judge_field,
                    oracle_field=oracle_field,
                )
            else:
                metadata = dict(getattr(raw_probe, "metadata", None) or {})

                def _probe_value(name: str, default: Any = None) -> Any:
                    if name.startswith("metadata."):
                        return metadata.get(name.split(".", 1)[1], default)
                    value = getattr(raw_probe, name, None)
                    return value if value is not None else metadata.get(name, default)

                object_record: Dict[str, Any] = {
                    "prompt_id": _probe_value("prompt_id"),
                    "prompt": _probe_value("prompt", ""),
                    "response": _probe_value("response"),
                    "judge_score": _probe_value(judge_field),
                    "oracle_label": _probe_value(oracle_field),
                    "metadata": metadata,
                }
                for name in (
                    "source_id",
                    "row_id",
                    "observation_id",
                    "sample_weight",
                    *covariate_names,
                ):
                    value = _probe_value(name)
                    if value is not None:
                        object_record[name] = value
                canonical = canonicalize_record(
                    object_record,
                    index,
                    source_id=f"transport_probe:{policy}",
                    policy=policy,
                )

            if canonical.get("oracle_label") is None:
                raise ValueError(
                    f"Transport probe {index} for policy {policy!r} is "
                    f"missing oracle field {oracle_field!r}."
                )
            source_row = (
                str(canonical["source_id"]),
                str(canonical["row_id"]),
            )
            observation_id = canonical.get("observation_id")
            if source_row in used_row_ids or (
                observation_id is not None
                and str(observation_id) in used_observation_ids
            ):
                raise ValueError(
                    f"Transport probe {index} for policy {policy!r} was also "
                    "used to fit the calibrator; probes must be held out."
                )

            oracle_value = float(canonical["oracle_label"])
            validate_values_on_scale(
                np.asarray([oracle_value], dtype=float),
                oracle_input_scale,
                field_name="transport probe oracle_label",
            )
            canonical["oracle_label"] = float(
                output_scale.inverse(oracle_input_scale.normalize(oracle_value))
            )

            if "response_length" in covariate_names:
                response = canonical.get("response")
                if response is None:
                    raise ValueError(
                        f"Transport probe {index} for policy {policy!r} is "
                        "missing 'response'; it is required to compute fitted "
                        "covariate 'response_length'."
                    )
                metadata = dict(canonical.get("metadata") or {})
                metadata["response_length"] = float(len(str(response).split()))
                canonical["metadata"] = metadata

            probes.append(canonical)

        assert config is not None
        delta_max = config.delta_max_by_policy.get(policy)
        if delta_max is None:
            # An explicit TransportAuditConfig is the new 0.6.0 API — not a
            # 0.5.x migration case — so the module-level FutureWarning about
            # the vocabulary change does not apply. Note the consequence
            # concisely instead; direct audit_transportability calls keep
            # the FutureWarning.
            logger.info(
                "Transport probe for policy %r has no delta_max margin: the "
                "audit is descriptive-only (NOT_GRADED).",
                policy,
            )
        with warnings.catch_warnings():
            if delta_max is None:
                warnings.simplefilter("ignore", FutureWarning)
            diagnostic = audit_transportability(
                results.calibrator,
                probes,
                bins=config.bins,
                group_label=f"policy:{policy}",
                alpha=config.alpha,
                delta_max=delta_max,
                family_size=config.resolved_family_size,
                min_effective_clusters=config.min_effective_clusters,
            )
        audits[policy] = {
            **diagnostic.to_dict(),
            "performed": True,
            "graded": diagnostic.status in {"PASS", "FAIL", "INCONCLUSIVE"},
            "applies_to_current_estimate": route_by_policy[policy] != "direct_oracle",
        }

        if diagnostic.status == "FAIL" and route_by_policy[policy] != "direct_oracle":
            gates = results.metadata.setdefault("reliability_gates", {})
            gate = gates.setdefault(
                policy,
                {
                    "flagged": False,
                    "refused": False,
                    "refuse_level_claims": False,
                    "reasons": [],
                },
            )
            gate["flagged"] = True
            gate["refuse_level_claims"] = True
            reasons = list(gate.get("reasons") or [])
            reason = (
                "residual transport FAIL: simultaneous CI "
                f"[{diagnostic.delta_ci[0]:+.3f}, "
                f"{diagnostic.delta_ci[1]:+.3f}] is outside margin "
                f"+/-{diagnostic.delta_max:.3f}"
            )
            if reason not in reasons:
                reasons.append(reason)
            gate["reasons"] = reasons
            if results.diagnostics is not None:
                # Only the audited-and-failed policy gets a status here;
                # never-assessed policies stay absent rather than being
                # recorded as GOOD. Consumers read status_per_policy with
                # .get()/items(), so missing keys are fine.
                statuses = dict(results.diagnostics.status_per_policy or {})
                statuses[policy] = Status.CRITICAL
                results.diagnostics.status_per_policy = statuses

    status_order = ("FAIL", "INCONCLUSIVE", "NOT_GRADED", "NOT_CHECKED", "PASS")
    observed = {str(audit["status"]) for audit in audits.values()}
    overall = next(
        (status for status in status_order if status in observed), "NOT_CHECKED"
    )
    results.metadata["transport_audits"] = audits
    results.metadata["transport_status"] = overall
    if results.diagnostics is not None:
        results.diagnostics.transport_audits = audits
        results.diagnostics.transport_status_per_policy = {
            policy: str(audit["status"]) for policy, audit in audits.items()
        }


def _build_label_design(
    kind: str,
    propensities: Optional[Dict[str, Any]],
    fresh_draws_dict: Dict[str, Any],
    label_design_cls: Any,
) -> Any:
    """Validate the high-level labeling declaration for statistical core."""
    normalized_kind = str(kind).strip().lower()
    allowed = {"representative", "known_propensity", "targeted_unknown"}
    if normalized_kind not in allowed:
        raise ValueError(f"label_design must be one of {sorted(allowed)}, got {kind!r}")
    normalized_propensities: Optional[Dict[str, np.ndarray]] = None
    if normalized_kind == "known_propensity":
        if not propensities:
            raise ValueError(
                "label_design='known_propensity' requires label_propensities "
                "for every evaluation policy."
            )
        normalized_propensities = {}
        for policy, dataset in fresh_draws_dict.items():
            if policy not in propensities:
                raise ValueError(f"Missing label propensities for policy '{policy}'.")
            values = np.asarray(propensities[policy], dtype=float)
            if values.shape != (len(dataset.samples),):
                raise ValueError(
                    f"label_propensities['{policy}'] must have length "
                    f"{len(dataset.samples)}, got shape {values.shape}."
                )
            if (
                not np.all(np.isfinite(values))
                or np.any(values <= 0)
                or np.any(values > 1)
            ):
                raise ValueError(
                    f"label_propensities['{policy}'] must be finite values in "
                    "(0, 1]."
                )
            normalized_propensities[policy] = values
        extra = sorted(set(propensities) - set(fresh_draws_dict))
        if extra:
            raise ValueError(
                f"label_propensities contains unknown policies: {', '.join(extra)}"
            )
    elif propensities is not None:
        raise ValueError(
            "label_propensities is only valid with " "label_design='known_propensity'."
        )
    return label_design_cls(kind=normalized_kind, propensities=normalized_propensities)


def _build_calibration_provenance(
    dataset: Dataset,
    covariate_names: Optional[List[str]],
    provenance_cls: Any,
) -> Any:
    """Capture the exact labeled rows used by ``calibrate_dataset`` in order."""
    judge_scores: List[float] = []
    oracle_labels: List[float] = []
    prompt_ids: List[str] = []
    row_roles: List[str] = []
    evaluation_keys: List[Optional[Tuple[str, str, int]]] = []
    covariate_rows: List[List[float]] = []

    for sample in dataset.samples:
        if sample.oracle_label is None:
            continue
        if sample.judge_score is None:
            raise ValueError(
                f"Calibration row {sample.row_id!r} has an oracle label but no "
                "judge score."
            )
        judge_scores.append(float(sample.judge_score))
        oracle_labels.append(float(sample.oracle_label))
        prompt_ids.append(sample.prompt_id)
        role = str(sample.metadata.get("row_role", "external"))
        if role not in {"external", "evaluation"}:
            raise ValueError(
                f"Unknown calibration row role {role!r} for row {sample.row_id!r}."
            )
        row_roles.append(role)
        raw_key = sample.metadata.get("evaluation_key")
        if role == "evaluation":
            if not isinstance(raw_key, (tuple, list)) or len(raw_key) != 3:
                raise ValueError(
                    "Evaluation calibration rows require an exact "
                    "(policy, prompt_id, draw_idx) key."
                )
            evaluation_keys.append((str(raw_key[0]), str(raw_key[1]), int(raw_key[2])))
        else:
            evaluation_keys.append(None)
        if covariate_names:
            covariate_rows.append(
                [float(sample.metadata[name]) for name in covariate_names]
            )

    covariates = np.asarray(covariate_rows, dtype=float) if covariate_names else None
    return provenance_cls(
        judge_scores=np.asarray(judge_scores, dtype=float),
        oracle_labels=np.asarray(oracle_labels, dtype=float),
        prompt_ids=prompt_ids,
        covariates=covariates,
        row_roles=row_roles,
        evaluation_keys=evaluation_keys,
    )


def _analysis_provenance_metadata(
    *,
    fresh_draws_dict: Dict[str, Any],
    calibration_dataset: Optional[Dataset],
    fresh_judge_scale: Any,
    fresh_oracle_scale: Any,
    calibration_judge_scale: Optional[Any],
    calibration_oracle_scale: Optional[Any],
    output_scale: Any,
    invalid_mode: str,
    covariate_names: Optional[List[str]],
) -> Dict[str, Any]:
    evaluation_rows = {
        policy: [sample.row_id for sample in dataset.samples]
        for policy, dataset in fresh_draws_dict.items()
    }
    fit_rows = []
    fit_roles: Dict[str, int] = {"external": 0, "evaluation": 0}
    if calibration_dataset is not None:
        for sample in calibration_dataset.samples:
            if sample.oracle_label is None:
                continue
            fit_rows.append(sample.row_id)
            role = str(sample.metadata.get("row_role", "external"))
            fit_roles[role] = fit_roles.get(role, 0) + 1
    return {
        "validation": {"mode": "full", "on_invalid": invalid_mode},
        "evaluation_rows": evaluation_rows,
        "calibration_fit_rows": fit_rows,
        "calibration_fit_roles": fit_roles,
        "covariates": list(covariate_names or []),
        "scales": {
            "fresh_judge": fresh_judge_scale.to_dict(),
            "fresh_oracle": fresh_oracle_scale.to_dict(),
            "calibration_judge": (
                calibration_judge_scale.to_dict()
                if calibration_judge_scale is not None
                else None
            ),
            "calibration_oracle": (
                calibration_oracle_scale.to_dict()
                if calibration_oracle_scale is not None
                else None
            ),
            "output": output_scale.to_dict(),
            "internal": {"min": 0.0, "max": 1.0, "is_identity": True},
        },
    }


def _mark_uncalibrated_result(
    results: EstimationResult,
    uncalibrated_policies: List[str],
    *,
    n_label_clusters: int = 0,
) -> None:
    """Grade judge-only estimates without hiding their descriptive value."""
    if n_label_clusters:
        reason = (
            f"Only {n_label_clusters} independent oracle-labeled prompt clusters "
            "were supplied; at least 4 are required for cross-fitted calibration. "
            "This is a raw judge-score mean, not an oracle-calibrated policy value."
        )
    else:
        reason = (
            "No oracle labels were supplied; this is a raw judge-score mean, "
            "not an oracle-calibrated policy value."
        )
    gates = results.metadata.setdefault("reliability_gates", {})
    for policy in uncalibrated_policies:
        gates[policy] = {
            "flagged": True,
            "refused": False,
            "refuse_level_claims": True,
            "claim_status": "UNCALIBRATED",
            "reasons": [reason],
        }
    if results.diagnostics is not None:
        from ..diagnostics import Status

        statuses = dict(results.diagnostics.status_per_policy or {})
        for policy in uncalibrated_policies:
            statuses[policy] = Status.WARNING
        results.diagnostics.status_per_policy = statuses


def _scale_oracle_source_metadata(metadata: Dict[str, Any], output_scale: Any) -> None:
    """Put cross-source oracle disagreements in the result's public units."""
    for conflict_key in ("conflicts", "value_conflicts"):
        conflicts = metadata.get(conflict_key)
        if not isinstance(conflicts, list):
            continue
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                continue
            for key in ("existing_value", "new_value"):
                value = conflict.get(key)
                if isinstance(value, (int, float)):
                    conflict[key] = float(output_scale.inverse(float(value)))
            difference = conflict.get("difference")
            if isinstance(difference, (int, float)):
                conflict["difference"] = float(difference) * output_scale.span
    metadata["oracle_value_scale"] = output_scale.to_dict()


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
                if (
                    not hasattr(sample, "response")
                    or sample.response is None
                    or sample.metadata.get("_cje_response_missing", False)
                ):
                    raise ValueError(
                        f"include_response_length=True requires all samples to have a 'response' field. "
                        f"Sample {i} (prompt_id={sample.prompt_id}) is missing this field or it is None."
                    )

        # Add response_length to the front of the list if not already there
        if "response_length" not in covariates:
            covariates.insert(0, "response_length")

    if dataset is not None:
        for covariate in covariates:
            for i, sample in enumerate(dataset.samples):
                if covariate == "response_length":
                    value = float(len(sample.response.split()))
                    sample.metadata[covariate] = value
                    continue
                if covariate not in sample.metadata:
                    raise ValueError(
                        f"Covariate '{covariate}' is missing for sample {i} "
                        f"(prompt_id={sample.prompt_id})."
                    )
                try:
                    value = float(sample.metadata[covariate])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Covariate '{covariate}' must be numeric for sample "
                        f"{i} (prompt_id={sample.prompt_id})."
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"Covariate '{covariate}' must be finite for sample "
                        f"{i} (prompt_id={sample.prompt_id})."
                    )
                sample.metadata[covariate] = value

    return covariates if covariates else None


def _denormalize_results(
    results: EstimationResult,
    *,
    output_scale: Any,
    judge_input_scale: Any,
    fresh_norm_info: Optional[Any],
    result_scale_source: str,
    result_estimand: str,
    verbose: bool,
) -> None:
    """Transform every public statistical artifact from internal units."""
    scale_factor = output_scale.max_val - output_scale.min_val
    results.estimates = output_scale.inverse_array(results.estimates)
    results.standard_errors = results.standard_errors * scale_factor

    # Paired bootstrap replicate matrix: replicates are estimates, so they
    # get the same affine inverse transform (compare_policies then returns
    # differences/SEs/CIs on the original scale)
    if results.bootstrap_samples is not None:
        results.bootstrap_samples = (
            results.bootstrap_samples * scale_factor + output_scale.min_val
        )

    if results.influence_functions:
        results.influence_functions = {
            policy: np.asarray(values, dtype=float) * scale_factor
            for policy, values in results.influence_functions.items()
        }

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
            for difference_key in ("difference", "ci_lower", "ci_upper"):
                value = pair_entry.get(difference_key)
                if isinstance(value, (int, float)):
                    pair_entry[difference_key] = float(value * scale_factor)

    se_components = results.metadata.get("se_components")
    if isinstance(se_components, dict):
        variances = se_components.get("oracle_variance_per_policy")
        if isinstance(variances, dict):
            se_components["oracle_variance_per_policy"] = {
                policy: float(value * scale_factor**2)
                for policy, value in variances.items()
                if isinstance(value, (int, float))
            }

    point_estimator = results.metadata.get("point_estimator")
    if isinstance(point_estimator, dict):
        plug_in = point_estimator.get("plug_in_estimates")
        if isinstance(plug_in, list):
            point_estimator["plug_in_estimates"] = [
                float(output_scale.inverse(float(value))) for value in plug_in
            ]
        corrections = point_estimator.get("residual_corrections")
        if isinstance(corrections, list):
            point_estimator["residual_corrections"] = [
                float(value) * scale_factor for value in corrections
            ]

    # Also denormalize bootstrap CIs if present (metadata mirror AND the
    # typed ci_info record must stay in sync)
    if "bootstrap_ci" in results.metadata:
        boot_ci = results.metadata["bootstrap_ci"]
        boot_ci["lower"] = [
            float(v * scale_factor + output_scale.min_val) for v in boot_ci["lower"]
        ]
        boot_ci["upper"] = [
            float(v * scale_factor + output_scale.min_val) for v in boot_ci["upper"]
        ]
        if results.ci_info is not None and results.ci_info.method == "percentile":
            results.ci_info.lower = list(boot_ci["lower"])
            results.ci_info.upper = list(boot_ci["upper"])

    diagnostics = results.diagnostics
    if diagnostics is not None:
        diagnostics.estimates = {
            policy: float(output_scale.inverse(value))
            for policy, value in diagnostics.estimates.items()
        }
        diagnostics.standard_errors = {
            policy: float(value * scale_factor)
            for policy, value in diagnostics.standard_errors.items()
        }
        if diagnostics.calibration_rmse is not None:
            diagnostics.calibration_rmse = float(
                diagnostics.calibration_rmse * scale_factor
            )
        if diagnostics.calibration_tolerance is not None:
            diagnostics.calibration_tolerance = float(
                diagnostics.calibration_tolerance * scale_factor
            )

    seen_cards = set()
    for cards in (
        getattr(diagnostics, "boundary_cards", None),
        results.metadata.get("boundary_cards"),
    ):
        if not isinstance(cards, dict) or id(cards) in seen_cards:
            continue
        seen_cards.add(id(cards))
        for card in cards.values():
            if not isinstance(card, dict):
                continue
            s_range = card.get("oracle_s_range")
            if isinstance(s_range, (list, tuple)) and len(s_range) == 2:
                converted = judge_input_scale.inverse_array(
                    np.asarray(s_range, dtype=float)
                )
                card["oracle_s_range"] = [float(converted[0]), float(converted[1])]
            if isinstance(card.get("partial_id_width"), (int, float)):
                card["partial_id_width"] = float(
                    card["partial_id_width"] * scale_factor
                )

    normalization = (
        fresh_norm_info.to_dict()
        if fresh_norm_info is not None
        else {
            "judge_score": {
                "original_range": (
                    judge_input_scale.min_val,
                    judge_input_scale.max_val,
                ),
                "is_identity": judge_input_scale.is_identity(),
                "origin": "unit_default",
            }
        }
    )
    normalization["output_scale"] = output_scale.to_dict()
    normalization["results_scale_source"] = result_scale_source
    # results_scale names the display axis; the estimand (what the numbers
    # mean) is labeled separately so a declared axis never relabels a raw
    # judge mean as an oracle quantity.
    normalization["results_scale"] = (
        "mixed_internal"
        if result_estimand == "mixed"
        else (
            "declared"
            if result_scale_source == "declared"
            else (
                "judge_original"
                if result_estimand == "judge_mean"
                else "oracle_original"
            )
        )
    )
    results.metadata["normalization"] = normalization
    if hasattr(results, "units"):
        from ..data.models import ResultUnits

        results.units = ResultUnits(
            estimand=result_estimand,
            output_scale=output_scale.to_dict(),
            judge_input_scale=judge_input_scale.to_dict(),
            internal_scale={"min": 0.0, "max": 1.0, "is_identity": True},
        )

    if verbose:
        logger.info(
            "Transformed result artifacts to %s scale [%s, %s]",
            result_scale_source,
            output_scale.min_val,
            output_scale.max_val,
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
    """Pool labeled response rows without value-based deduplication."""
    from ..data.models import Sample, _json_safe

    combined_samples: List[Sample] = []
    seen_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    oracle_by_observation: Dict[str, List[Tuple[str, float]]] = {}
    rows_by_prompt_score: Dict[
        Tuple[str, float], List[Tuple[str, str, Optional[str], float]]
    ] = {}
    conflicts: List[Dict[str, Any]] = []
    value_conflicts: List[Dict[str, Any]] = []
    n_duplicates = 0

    def _add_sample(
        sample: Any,
        *,
        source_family: str,
        fallback_index: int,
        evaluation_key: Optional[Tuple[str, str, int]] = None,
    ) -> bool:
        nonlocal n_duplicates
        if sample.oracle_label is None or sample.judge_score is None:
            return False
        if evaluation_key is not None:
            policy, prompt_id, draw_idx = evaluation_key
            fallback_source_id = f"fresh_draws:{policy}"
            fallback_row_id = f"{fallback_source_id}:prompt:{prompt_id}:draw:{draw_idx}"
            row_source = f"fresh_draws:{policy}:draw{draw_idx}"
        else:
            fallback_source_id = source_family
            fallback_row_id = f"{fallback_source_id}:row:{fallback_index}"
            row_source = source_family

        source_id = str(sample.source_id or fallback_source_id)
        row_id = str(sample.row_id or fallback_row_id)
        identity = (source_id, row_id)
        signature = {
            "prompt_id": str(sample.prompt_id),
            "response": sample.response,
            "judge_score": float(sample.judge_score),
            "oracle_label": float(sample.oracle_label),
            "observation_id": sample.observation_id,
            "metadata": _json_safe(dict(sample.metadata or {})),
        }
        previous = seen_rows.get(identity)
        if previous is not None:
            if previous != signature:
                raise ValueError(
                    f"Conflicting records share row_id {row_id!r} in source "
                    f"{source_id!r}. Row identity must be unique."
                )
            n_duplicates += 1
            return False
        seen_rows[identity] = signature

        observation_id = sample.observation_id
        if observation_id is not None:
            for prev_source, prev_oracle in oracle_by_observation.get(
                observation_id, []
            ):
                if (
                    prev_source != source_family
                    and abs(prev_oracle - float(sample.oracle_label)) > 0.05
                ):
                    conflicts.append(
                        {
                            "observation_id": observation_id,
                            "existing_value": float(prev_oracle),
                            "existing_source": prev_source,
                            "new_value": float(sample.oracle_label),
                            "new_source": source_family,
                            "difference": abs(prev_oracle - float(sample.oracle_label)),
                        }
                    )
                    break
            oracle_by_observation.setdefault(observation_id, []).append(
                (source_family, float(sample.oracle_label))
            )

        # Value-based cross-source conflict CHECK (never a merge): rows on the
        # same prompt with the same judge score but materially different
        # oracle labels usually describe the same response labeled twice.
        # Pairs where both rows carry explicit observation_ids are governed by
        # the identity check above (distinct ids assert distinct observations).
        value_key = (str(sample.prompt_id), float(sample.judge_score))
        for (
            prev_source,
            prev_row_id,
            prev_obs_id,
            prev_oracle,
        ) in rows_by_prompt_score.get(value_key, []):
            if (
                prev_source != source_family
                and (prev_obs_id is None or observation_id is None)
                and abs(prev_oracle - float(sample.oracle_label)) > 0.05
            ):
                value_conflicts.append(
                    {
                        "prompt_id": value_key[0],
                        "judge_score": value_key[1],
                        "existing_source": prev_source,
                        "existing_row_id": prev_row_id,
                        "existing_value": float(prev_oracle),
                        "new_source": source_family,
                        "new_row_id": row_id,
                        "new_value": float(sample.oracle_label),
                        "difference": abs(prev_oracle - float(sample.oracle_label)),
                    }
                )
                break
        rows_by_prompt_score.setdefault(value_key, []).append(
            (source_family, row_id, observation_id, float(sample.oracle_label))
        )

        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "source": row_source,
                "row_role": (
                    "evaluation" if source_family == "fresh_draws" else "external"
                ),
            }
        )
        if evaluation_key is not None:
            metadata["evaluation_key"] = evaluation_key
        combined_samples.append(
            Sample(
                prompt_id=str(sample.prompt_id),
                prompt=getattr(sample, "prompt", "") or "",
                response=getattr(sample, "response", "") or "",
                reward=None,
                judge_score=float(sample.judge_score),
                oracle_label=float(sample.oracle_label),
                metadata=metadata,
                row_id=row_id,
                observation_id=observation_id,
                source_id=source_id,
            )
        )
        return True

    n_from_logged = 0
    if logged_dataset:
        for index, sample in enumerate(logged_dataset.samples):
            if _add_sample(sample, source_family="logged_data", fallback_index=index):
                n_from_logged += 1

    n_from_fresh = 0
    if fresh_draws_per_policy:
        for policy, fd_dataset in fresh_draws_per_policy.items():
            for index, fd_sample in enumerate(fd_dataset.samples):
                if _add_sample(
                    fd_sample,
                    source_family="fresh_draws",
                    fallback_index=index,
                    evaluation_key=(
                        str(policy),
                        str(fd_sample.prompt_id),
                        int(fd_sample.draw_idx),
                    ),
                ):
                    n_from_fresh += 1

    n_from_calib = 0
    if calibration_dataset:
        for index, sample in enumerate(calibration_dataset.samples):
            if _add_sample(
                sample, source_family="calibration_data", fallback_index=index
            ):
                n_from_calib += 1

    if conflicts:
        logger.warning(
            "Found %d explicit response identities with conflicting oracle "
            "labels across sources. All rows are retained; inspect "
            "oracle_sources.conflicts.",
            len(conflicts),
        )

    if value_conflicts:
        preview = "; ".join(
            f"prompt_id={conflict['prompt_id']!r} "
            f"judge_score={conflict['judge_score']:g}: "
            f"{conflict['existing_source']}:{conflict['existing_row_id']}"
            f"={conflict['existing_value']:.3f} vs "
            f"{conflict['new_source']}:{conflict['new_row_id']}"
            f"={conflict['new_value']:.3f}"
            for conflict in value_conflicts[:5]
        )
        logger.warning(
            "Found %d prompt/judge-score pairs with materially different "
            "oracle labels across sources and no shared observation_id. All "
            "rows are retained, NOT merged; set observation_id to assert "
            "shared response identity. Inspect oracle_sources.value_conflicts. "
            "Pairs: %s",
            len(value_conflicts),
            preview,
        )

    if verbose:
        logger.info(
            "Combined oracle sources: %d total rows "
            "(calib=%d, fresh=%d, logged=%d, duplicate_rows=%d)",
            len(combined_samples),
            n_from_calib,
            n_from_fresh,
            n_from_logged,
            n_duplicates,
        )

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
        "fresh_draws": {"n_oracle": n_from_fresh, "coverage": None},
        "total_oracle": len(combined_samples),
        "n_conflicts": len(conflicts),
        "n_value_conflicts": len(value_conflicts),
        "n_duplicates": n_duplicates,
        "deduplication_key": "(source_id, row_id)",
    }
    if conflicts:
        oracle_sources_metadata["conflicts"] = conflicts[:10]
    if value_conflicts:
        oracle_sources_metadata["value_conflicts"] = value_conflicts[:10]

    if combined_samples:
        combined_dataset = Dataset(
            samples=combined_samples,
            target_policies=target_policies,
            metadata={
                "combined_oracle_sources": True,
                "deduplication_key": "(source_id, row_id)",
            },
        )
    elif calibration_dataset is not None:
        # Keep a nonempty container so the high-level pipeline can grade an
        # external zero-label file as uncalibrated instead of failing Dataset's
        # nonempty invariant. No row in this fallback is used for fitting.
        combined_dataset = calibration_dataset.model_copy(deep=True)
        combined_dataset.metadata.update(
            {
                "combined_oracle_sources": True,
                "deduplication_key": "(source_id, row_id)",
                "contains_no_oracle_pairs": True,
            }
        )
    else:
        raise ValueError("Cannot combine oracle sources: no oracle-labeled rows found")
    return combined_dataset, oracle_sources_metadata

"""Hydra entrypoint for CJE analysis.

Usage examples:
    python -m cje.interface.hydra_entry dataset=data.jsonl \
        estimator=dr-cpo fresh_draws_dir=responses/ estimator_config.n_folds=10

Hydra is configured to NOT change the working directory.
"""

from typing import Any, Dict, Optional
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from .analysis import analyze_dataset

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    # Convert nested config to primitives where necessary
    estimator_config: Dict[str, Any] = {}
    if "estimator_config" in cfg and cfg.estimator_config is not None:
        estimator_config = OmegaConf.to_container(cfg.estimator_config, resolve=True)  # type: ignore
        assert isinstance(estimator_config, dict)

    dataset: str = cfg.dataset
    estimator_name: str = cfg.get("estimator_name", "auto")
    judge_field: str = cfg.get("judge_field", "judge_score")
    oracle_field: str = cfg.get("oracle_field", "oracle_label")
    fresh_draws_dir: Optional[str] = cfg.get("fresh_draws_dir")
    verbose: bool = bool(cfg.get("verbose", False))
    output: Optional[str] = cfg.get("output")

    # Auto-default: stacked-dr if fresh draws specified, otherwise calibrated-ips
    if estimator_name in (None, "", "auto"):
        estimator = "stacked-dr" if fresh_draws_dir else "calibrated-ips"
    else:
        estimator = estimator_name

    logger.info(f"Running CJE via Hydra | estimator={estimator}")

    results = analyze_dataset(
        logged_data_path=dataset,
        estimator=estimator,
        judge_field=judge_field,
        oracle_field=oracle_field,
        estimator_config=estimator_config,
        fresh_draws_dir=fresh_draws_dir,
        verbose=verbose,
    )

    # Print a concise summary
    policies = results.metadata.get("target_policies", [])
    if policies:
        print("Results:")
        for i, p in enumerate(policies):
            est = results.estimates[i]
            se = results.standard_errors[i]
            print(f"  {p}: {est:.3f} ± {se:.3f}")

    # Optional export
    if output:
        try:
            from ..utils.export import export_results_json

            export_results_json(results, output)
            print(f"\n✓ Results saved to: {output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

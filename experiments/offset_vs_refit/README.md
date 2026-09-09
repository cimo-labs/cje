# Offset vs Refit Simulation

This experiment quantifies when offset-style corrections are enough, versus when you need to refit calibration.

It focuses on policy first moments:

- Target: `V(policy) = E[Y | policy]`
- Legacy estimate: `E[f_old(S)]`
- Offset estimate: `E[f_old(S)] + delta_hat`, where `delta_hat = E_audit[Y - f_old(S)]`
- Policy-offset estimate (EIF-like first moment): `E[f_old(S)] + E_audit,policy[Y - f_old(S)]`
- Refit estimate: learn `f_new` from recent audit labels (or pooled labels), then use `E[f_new(S)]`

## Scenarios

The simulation runs four drift scenarios:

1. `intercept_shift`
2. `slope_shift`
3. `nonlinear_shift`
4. `covariate_interaction_shift`

And two audit-slice profiles:

1. `base_heavy` (a probability sample of a mixture dominated by the base policy)
2. `balanced` (equal policy weights in the sampling mixture)

Both profiles sample randomly within policies. Their pooled residuals target
different policy mixtures: neither pooled audit is a per-policy transport check.
Policy offsets use within-policy residuals; when a policy receives no audit labels,
that comparison baseline falls back to the global correction.

## Methods Compared

1. `old_plugin`
2. `old_plus_global_offset`
3. `old_plus_policy_offset` (policy-specific residual correction)
4. `recent_refit_monotone`
5. `pooled_refit_monotone`
6. `recent_refit_two_stage`
7. `pooled_refit_two_stage`

## Run

From the repository root, install the optional research dependencies (`pandas`)
and plotting extra, then predeclare a practical mean-residual margin:

```bash
poetry install --with research --extras viz
poetry run python experiments/offset_vs_refit/offset_vs_refit_simulation.py \
  --delta-max 0.05 \
  --audit-sizes 20,50,100,200 \
  --n-reps 60 \
  --output-dir experiments/offset_vs_refit/results
```

Faster smoke run:

```bash
poetry run python experiments/offset_vs_refit/offset_vs_refit_simulation.py \
  --delta-max 0.05 \
  --n-reps 8 \
  --audit-sizes 20,50 \
  --no-plots
```

Run experiment tests:

```bash
poetry run pytest -q experiments/offset_vs_refit/test_offset_vs_refit_simulation.py
```

`--delta-max` is required. The example's `0.05` is an illustrative five-point
margin on the synthetic [0, 1] oracle scale, chosen before examining residuals;
replace it with the tolerance relevant to the decision being studied. The Python
`run_experiment_suite` function uses that same illustrative default and accepts
an explicit `delta_max`. Use `--audit-alpha` for the family-wise error level.

The default audit family includes every scenario/profile/audit-size cell within
one replicate (32 for the full command above). `--family-size` can declare a larger
decision family; it cannot undercount the configured cells. Monte Carlo replicates
are separate simulated worlds for estimating rates. The seven correction methods
reuse the same old-calibrator audit and do not count as seven independent audits.

## Outputs

Written to `--output-dir`:

1. `offset_vs_refit_raw.csv`: replicate-level metrics
2. `offset_vs_refit_summary.csv`: aggregated metrics
3. `method_mae_at_max_audit.png`: method ranking at largest audit size
4. `offset_vs_refit_by_audit_size.png`: global offset vs policy offset vs refit as audit size grows
5. `run_config.json`: reproducibility config

## Key Metrics

1. `mae_policy_mean`: mean absolute bias across policies
2. `rmse_policy_mean`: root mean squared policy-mean error
3. `ranking_accuracy`: best-policy selection accuracy
4. `transport_status`: PASS/FAIL/INCONCLUSIVE for the **old calibrator** on the
   independent audit slice, graded against the predeclared practical margin
5. `transport_delta_max`, `transport_alpha`, `transport_family_size`, and
   `transport_effective_clusters`: the audit design and effective sample size

`PASS` requires the simultaneous residual CI to lie wholly within the margin and
at least 20 effective independent clusters. A wholly out-of-margin CI is `FAIL`,
including below that cluster floor; boundary overlap or an unresolved small probe
is `INCONCLUSIVE`. The summary exports rates for the full current status vocabulary,
including `NOT_GRADED` and `NOT_CHECKED`; those two rates are zero in these runs
because every cell supplies a margin and probes. Legacy `WARN` is not emitted.

Old-fit, audit, and evaluation prompt IDs are disjoint. Calibration uses `fit_cv`
with observed labels and prompt clusters. The audit is recorded before its labels
are used for residual corrections or recent/pooled refits. Those refits are evaluated
on separate synthetic samples but are **not transport-audited using their own fit
labels**. The shared `transport_status` column must not be read as certifying them.

The study measures errors against synthetic evaluation means and best-policy
selection rates. It does not establish confidence-interval coverage, achieved
statistical power, or transport to a user's population. `run_config.json` preserves
the margin, alpha, family, population interpretation, and those limitations.

## Expected Pattern

1. `intercept_shift`: policy offset should be competitive with recent refit.
2. `slope/nonlinear drift`: policy offset should clearly beat global offset.
3. `covariate_interaction_shift`: two-stage recent refit should usually win.
4. Treat `old_plus_global_offset` as a baseline, not as the default production correction.

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

1. `base_heavy` (audit not representative)
2. `balanced` (more representative)

## Methods Compared

1. `old_plugin`
2. `old_plus_global_offset`
3. `old_plus_policy_offset` (policy-specific residual correction)
4. `recent_refit_monotone`
5. `pooled_refit_monotone`
6. `recent_refit_two_stage`
7. `pooled_refit_two_stage`

## Run

From repo root:

```bash
cd CJE/cje
python -m cje.experiments.offset_vs_refit_simulation \
  --audit-sizes 20,50,100,200 \
  --n-reps 60 \
  --output-dir cje/experiments/offset_vs_refit_results
```

Faster smoke run:

```bash
python -m cje.experiments.offset_vs_refit_simulation \
  --n-reps 8 \
  --audit-sizes 20,50 \
  --no-plots
```

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
4. `transport_status`: PASS/WARN/FAIL from transport audit on audit slice

## Expected Pattern

1. `intercept_shift`: policy offset should be competitive with recent refit.
2. `slope/nonlinear drift`: policy offset should clearly beat global offset.
3. `covariate_interaction_shift`: two-stage recent refit should usually win.
4. Treat `old_plus_global_offset` as a baseline, not as the default production correction.

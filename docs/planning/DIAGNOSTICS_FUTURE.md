# Diagnostics: Future Plan (Parsimony, Reliability, Usability)

This document proposes a pragmatic refactor of the diagnostics suite to deliver a concise, decision‑first default while preserving our current depth for audits and research.

## Objectives
- Parsimony: Default to a compact “Core 5” diagnostics set; move detail to drill‑down.
- Reliability: Keep orthogonal, high‑signal checks that map to distinct failure modes.
- Usability: One‑line state per policy, clear ship/stop, and 1–2 actionable suggestions.
- Auditability: Compute everything under the hood; simple by default, deep on demand.

## Core 5 Diagnostics (default view)
1) Overlap: Hellinger affinity (structural) + aESS/IFR (statistical; IFR_OUA when available).
2) Tails: Hill tail index (α < 2 critical; < 1 extreme).
3) Calibration Floor: f_min = min f(S) on oracle slice; floor_mass (logged/DR): fraction with f(S_eval) ≈ f_min; low‑S label coverage (bottom S deciles); flags bias risk even when S‑range is covered.
4) DR Validity: Orthogonality CI covers 0 (E[W·(R − q̂)] ≈ 0).
5) Roll‑up: CF‑bits bits_tot with dominant component (identification vs sampling).

## Simple Gates (proposed defaults)
- REFUSE: Hellinger < 0.20 OR aESS < 0.05 OR tail < 1.0 OR floor_mass ≥ 0.25.
- CRITICAL: aESS < 0.20 OR tail < 2.0.
- WARNING: IFR < 0.5 OR floor_mass ∈ [0.10, 0.25) OR low‑S label coverage < 5%.
- GOOD: none of the above AND DR orthogonality CI covers 0.

Each gate maps to 1–2 suggestions (fresh draws, low‑S labels, tighten ρ, trim tails). OUA jackknife remains optional (off by default) but supported everywhere.

## Instrumentation (Phase 1)
Compute and attach minimal metrics to `EstimationResult.metadata` (no UI change):
- Overlap: Hellinger, aESS/IFR, tail index, mass concentration (existing).
- Calibration Floor (new): `f_min`, `floor_mass_logged`, `floor_mass_fresh` (DR), low‑S label coverage (e.g., P(L=1|S≤q10,q20)), isotonic level count, slope in bottom 20%.
- DR: Orthogonality CI, DM–IPS split, IF tails (existing).
- CF‑bits: bits_tot, w_id, w_var, dominant (existing); include IFR_OUA when jackknife enabled.
- OUA (optional): `robust_standard_errors`, and metadata.oua = {var_oracle_per_policy, jackknife_counts} (now implemented).

Persist a flat record per policy/run for offline analysis (CSV/Parquet) via a lightweight aggregator script.

## Empirical Validation (Phase 2)
- Redundancy: Correlation/MI of metrics; PCA to estimate signal dimension; expect Hellinger vs aESS/IFR vs tails vs calibration floor to be distinct.
- Predictiveness: Define “should not ship” labels from current gates (overlap/tails/refusal) + DR invalidity + calibration floor WARN/CRIT. Train simple threshold stumps/small tree to predict labels from Core 5; compare AUROC/PR vs full metric set.
- Threshold tuning: Grid search thresholds to minimize 10× false‑ship + 1× false‑refuse. Use CV across policies/runs; report stability.
- Action value: Use CF‑bits widths to estimate expected bits gain per action (add K fresh draws, add M low‑S labels, tighten ρ). Rank suggestions by gain per unit cost.

Deliverable: a short internal note with confusion matrices, tuned thresholds, and recommended defaults.

## UX & API (Phase 3)
- Default “simple” view: Core 5 + one‑line state and 1–2 suggestions. Example: 
  - “Overlap good (A=0.52 | aESS=0.42). Tails ok (α=2.6). Cal floor WARN (floor 12%, f_min=0.40; low‑S labels 3%). DR orthog ✓. CF‑bits 1.8 (sampling‑dom).”
- Drill‑down on fail: Show DM–IPS split, IF tails, augmentation/OUA shares, coverage/floor plots.
- Config: `diagnostic_level: simple|full` (default simple). Everything remains available under `metadata.advanced` for audits.

## Rollout & Timeline
- Phase 1 (instrument): Add calibration‑floor metrics and per‑policy `metadata.core` summary. Keep current tests/outputs intact.
- Phase 2 (analyze): Run across Arena and internal datasets; tune thresholds; propose defaults.
- Phase 3 (flip default): Enable “simple” UI, Core 5 gates; keep full diagnostics behind toggle.
- Phase 4 (maintenance): Quarterly recalibration; add dependence‑robust SEs and drift checks (rank‑drift) as advanced tools.

## Risks & Mitigations
- Threshold sensitivity: Start WARN‑only; tune after evidence collection.
- Hiding useful signals: Keep “full” mode; log all metrics; retain auditability.
- OUA compute: Jackknife remains off by default; enable for high‑stakes runs.

## Open Questions
- Should calibration‑floor WARN escalate to CRITICAL for certain KPIs/domains?
- Where to set the epsilon for `floor_mass` (e.g., |f(S)−f_min| ≤ 1e‑3)?
- How to present dependence‑robust SEs (cluster/block bootstrap) by default for time‑ordered logs?

---

This plan lets us be evidence‑driven: instrument now, validate Core 5 predictiveness, then flip to a calm default that preserves safety and auditability.

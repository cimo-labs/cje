# Label-selection sensitivity on a fixed Arena pool

This offline example uses 600 prompts sampled with seed `20261022` from the 5,000-prompt overlap between the two policies. The derived `data.jsonl` contains only four fields and no response text; its SHA-256 is `3d515c20818bc60e152b9631375d293f5a3a0dcf5bfbfdf5f06aa931888a9330`. GPT-4.1-nano supplied judge scores; GPT-5 supplied oracle labels (not human ground truth). Source metadata calls GPT-5 the judge; see `provenance.json` for pinned source commits and source-file hashes.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python examples/label_selection/run.py
# Longer diagnostic run:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python examples/label_selection/run.py --repetitions 200
```

The notebook invokes the same runner. Dependencies are CJE's existing NumPy/SciPy/scikit-learn/Pydantic requirements. No model calls or internet access are needed.

## Frozen design

Equal-count judge-score bins contain 120 rows each, with stable tie splitting. The symmetric flag takes the 36 lowest oracle-minus-judge residuals per bin in each policy. The new differential condition flags 60 per bin for base and 24 for premium (50% versus 20%). These counts were fixed before the full run; they were not searched to produce a reversal. The fully labeled pool says base is the worse policy.

Flagged-only samples take 12 flagged rows per bin. Known-propensity samples take 9 flagged plus 3 unflagged per bin, recording the exact positive inclusion probabilities for every row. This changes both support and weighting; recovery cannot be attributed to weighting alone. It is an oracle-assisted best case, not a practical pre-label acquisition method or observed failure-report behavior.

The two symmetric flagged analyses share exactly the same sampled masks. Other conditions use explicit SHA256-derived random streams. Old pilot outputs are preserved separately; this revision uses a new seed namespace.

## Interpretation

Coverage is conditional on this fixed prompt pool and varies label selection only. Mean CI half-width is reported alongside bias, RMSE, and coverage. The intervals may include variation beyond the fixed-pool repetition design; high coverage alone is not a proof of correct population inference.

The low flagged-only paired-difference RMSE in the original pilot reflected near-identical label sets across repetitions, not general precision. In this revision, each symmetric flagged sample takes 12 of 36 flagged rows per bin, and the output reports distinct masks. In the targeted_unknown route, calibration can also yield low variability while transport remains explicitly unverified.

The score-quintile chi-square is a custom diagnostic, not a claimed built-in CJE gate. It uses score-value quantiles, preserves ties, and collapses duplicate edges. A p-value >= .05 is failure to detect unequal score-bin label rates, not proof that outcome-dependent selection is absent. Every contingency table, chi-square statistic, degrees of freedom, p-value, sampled-label count, and paired-comparison diagnostic is retained in `results/runs.json` when you run it.

Opposite-sign fraction is a descriptive sign change of the estimated policy difference, not a statistically significant ranking claim. No training or deployment conclusion follows.

Implementation and documentation are AI-assisted. See RESULTS.md for measured output and runtime.

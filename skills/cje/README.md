# CJE agent skill

Teaches a coding agent to run [CJE](https://github.com/cimo-labs/cje) correctly — plan a future
evaluation from pilot data, reshape existing eval data, drive the labeling loop, calibrate,
compare policies, and preserve results and refusal gates in a reproducible audit bundle.
Planning is optional; its power estimates are projections under explicit assumptions.

[`SKILL.md`](SKILL.md) is the entry point; [`reference.md`](reference.md) holds the full API detail and loads on demand. Both are plain Markdown and agent-agnostic.

## Install

Install on Python 3.10–3.13. Run these commands with the interpreter that will analyze the
data so the instructions match the installed package.

```bash
python -m pip install -U "cje-eval>=0.10.1"
python -m cje --version
python -m cje skill               # prints SKILL.md
python -m cje skill --reference   # prints reference.md
python -m cje skill --path        # prints the folder holding both
```

**Agents with a skills directory** (Claude Code and compatible): copy both files into it.

```bash
# For one project use .claude/skills/cje instead.
skill_dir=~/.claude/skills/cje
mkdir -p "$skill_dir"
python -m cje skill > "$skill_dir/SKILL.md"
python -m cje skill --reference > "$skill_dir/reference.md"
```

**Any other agent**: install the package as above (or let the agent do it), then paste this into the conversation, filling in the brackets:

```text
Record `python -m cje --version`. Read the full `python -m cje skill` and
`python -m cje skill --reference` from that interpreter and follow the skill's stop checks.
Use [file paths or in-memory records]. Compare [candidate minus baseline] on [outcome and scale].
The target frame is [eligible prompts or documents, including any missing scores or labels].
Historical calibration labels were selected by [mechanism and known rates, if used].
Current traffic and human reviews were selected by [mechanism and known rates at each stage,
or unknown]. Our prespecified decision margin is [gap in outcome units, or none].
```

For a worked example, follow the guide that matches how the data were collected.
[Weighted paired audits](../../guides/weighted-paired-audit.md) ·
[Binary judge review budgets](../../guides/evaluation-planning.md#plan-reviews-of-a-binary-judge) ·
[Rater panels and missing outcomes](../../guides/rater-and-missing-outcomes.md)

## Plan an evaluation

```text
Use the CJE skill to plan my next model comparison. Start from my pilot, target effect,
desired power, and labeling budget. Save the plan and assumptions, then produce a
reproducible comparison with confidence intervals and diagnostics when the data are ready.
```

For an executable pilot → plan → comparison example, use
[`scripts/planning_example.py`](scripts/planning_example.py) from a repository checkout
with the bundled data. It is optional demonstration material, not required for installation.

After installing the checkout (`pip install -e .`), run:

```bash
python skills/cje/scripts/planning_example.py --output-dir outputs/cje-planning
```

The example masks and reveals existing model-reference labels within a fixed cached-label
frame; it collects no new labels and makes no API calls. Inspect `planning.json` for the
budget/MDE tradeoff, `audit.json` for the paired comparison and unresolved transport checks,
and `console.log` for the readout. The directory also retains numeric inputs, sample IDs,
source hashes, settings, versions, warnings, and `rerun.sh`. This is a worked workflow, not
an empirical power or coverage validation.

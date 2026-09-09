# CJE agent skill

Teaches a coding agent to run [CJE](https://github.com/cimo-labs/cje) correctly — plan a future
evaluation from pilot data, reshape existing eval data, drive the labeling loop, calibrate,
compare policies, and preserve results and refusal gates in a reproducible audit bundle.
Planning is optional; its power estimates are projections under explicit assumptions.

[`SKILL.md`](SKILL.md) is the entry point; [`reference.md`](reference.md) holds the full API detail and loads on demand. Both are plain Markdown and agent-agnostic.

## Install

**Agents with a skills directory** (Claude Code and compatible): copy both files into it.

```bash
# Claude Code, all projects
mkdir -p ~/.claude/skills/cje
curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md -o ~/.claude/skills/cje/SKILL.md
curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/reference.md -o ~/.claude/skills/cje/reference.md

# Claude Code, one project (from a checkout of this repo)
mkdir -p .claude/skills
cp -r skills/cje .claude/skills/
```

**Any other agent**: no install needed — paste this into the conversation:

```text
Read https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md,
then use CJE to compare the policies in my eval data.
```

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

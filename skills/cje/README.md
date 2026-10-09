# CJE agent skill

Teaches a coding agent to run [CJE](https://github.com/cimo-labs/cje) correctly — plan a future
evaluation from pilot data, reshape existing eval data, drive the labeling loop, calibrate,
compare policies, and preserve results and refusal gates in a reproducible audit bundle.
Planning is optional; its power estimates are projections under explicit assumptions.

[`SKILL.md`](SKILL.md) is the entry point; [`reference.md`](reference.md) holds the full API detail and loads on demand. Both are plain Markdown and agent-agnostic.

## Install

The package ships the skill matching its own version (0.9.2+). Install on Python 3.10–3.13; on
Python 3.9 a bare `pip install cje-eval` silently installs the legacy 0.5 line, so keep the
version floor:

```bash
pip install -U "cje-eval>=0.9.2"
cje skill               # prints SKILL.md
cje skill --reference   # prints reference.md
cje skill --path        # prints the folder holding both
```

**Agents with a skills directory** (Claude Code and compatible): copy both files into it.

```bash
# Claude Code, all projects, from the installed package (0.9.2+)
mkdir -p ~/.claude/skills/cje
cp "$(cje skill --path)"/SKILL.md "$(cje skill --path)"/reference.md ~/.claude/skills/cje/

# Or from GitHub main (raw text)
curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md -o ~/.claude/skills/cje/SKILL.md
curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/reference.md -o ~/.claude/skills/cje/reference.md

# Claude Code, one project (from a checkout of this repo)
mkdir -p .claude/skills
cp -r skills/cje .claude/skills/
```

**Any other agent**: install the package as above (or let the agent do it), then paste this into the conversation, filling in the brackets:

```text
Run `cje skill` and follow it; when it points to reference.md, run `cje skill --reference`.
Data: [path to the eval export]
Question: [e.g. is candidate better than production on support prompts, and by how much?]
How the labels were chosen: [e.g. 50 random responses per policy rated by our QA team /
escalated tickets only / not sure]
```

If the agent cannot run `cje` (or has a version before 0.9.2), have it download the raw
[SKILL.md](https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md) and
[reference.md](https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/reference.md)
with `curl` and read the text itself; a summarizing web fetcher drops the rules.

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

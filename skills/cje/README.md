# CJE agent skill

Teaches a coding agent to run [CJE](https://github.com/cimo-labs/cje) correctly — reshape eval data, drive the labeling loop, calibrate, compare policies, and respect the refusal gates — instead of averaging raw judge scores.

[`SKILL.md`](SKILL.md) is the entry point; [`reference.md`](reference.md) holds the full API detail and loads on demand. Both are plain Markdown and agent-agnostic.

## Install

**Agents with a skills directory** (Claude Code and compatible): copy both files into it.

```bash
# Claude Code, all projects
mkdir -p ~/.claude/skills/cje
curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md -o ~/.claude/skills/cje/SKILL.md
curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/reference.md -o ~/.claude/skills/cje/reference.md

# Claude Code, one project (from a checkout of this repo)
cp -r skills/cje .claude/skills/
```

**Any other agent**: no install needed — paste this into the conversation:

```text
Read https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md,
then use CJE to compare the policies in my eval data.
```

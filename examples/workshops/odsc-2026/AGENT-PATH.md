# Drive the lab with your coding agent


This page is optional. The notebook is the main path, and the room moves with it. Use this page if you'd rather have your coding agent run each step. Make your prediction before you ask the agent: the exercises only work if your guess comes first. Use a frontier-size model, not a small fast one: in my checks, frontier-size models matched the notebook, while a small fast model got the numbers right but typed some results by hand and misexplained the exercises.

## Setup

This needs a local copy (the README shows how to get one; on Colab, follow the notebook instead). Open this folder in your agent and use Python 3.10-3.13. The README covers the 3.9 and 3.14 traps.

```sh
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r lab/requirements.txt
```

Paste this once:

```text
We're doing the ODSC CJE lab in this folder. The notebook's code is the reference design.
1. Use .venv's Python if present. If .venv does not exist or lacks cje-eval==0.9.3, create .venv
   with a Python 3.10-3.13 interpreter (`uv venv -p 3.12 .venv` or `python3.12 -m venv .venv`, or
   `python3 -m venv .venv` if `python3` is 3.10-3.13) and run `.venv/bin/pip install -r lab/requirements.txt`.
   If no Python 3.10-3.13 exists on this machine, stop and tell me to open the Colab link in README.md.
   Print the active Python version and cje-eval version (must be 0.9.3).
2. Run `cje skill > cje_skill.md` and `cje skill --reference > cje_reference.md` (use
   `python -m cje` if `cje` isn't on PATH). Read both files in full, and follow their hard rules
   and reporting order.
3. Run this once. It writes the notebook's code cells, each headed by its section, to
   agent-runs/notebook_code.py, without the facilitator's answer key:
   python -c "import json, os; C = json.load(open('workshop.ipynb', encoding='utf-8'))['cells']; S = lambda c: ''.join(c['source']); H = lambda i: next(S(C[j]).splitlines()[0].lstrip('# ') for j in range(i, -1, -1) if C[j]['cell_type'] == 'markdown' and S(C[j]).startswith('#')); os.makedirs('agent-runs', exist_ok=True); open('agent-runs/notebook_code.py', 'w', encoding='utf-8').write(''.join(f'# ==== {H(i)}\n{S(c)}\n\n' for i, c in enumerate(C) if c['cell_type'] == 'code' and 'PANEL_PATH' not in S(c)))"
   Work only from that file. Don't open workshop.ipynb, workshop-executed.*, facilitator/ or
   AGENT-PATH.md, and don't look in figures/: they hold the answers.
4. "Run the cells through section N" means: write agent-runs/<step>.py holding every cell of
   notebook_code.py from the top through that section's, unchanged and in order (later cells use
   names earlier ones define), then only the prints a prompt asks for. Stop after that cell:
   Exercise A′ is a separate cell after section 4, not part of it. Run it with
   MPLBACKEND=Agg and save stdout and stderr to agent-runs/<step>.txt.
   The cells write PNGs to figures/; let them. matplotlib's "non-interactive" notice is not a CJE
   warning. Quote only what that .txt prints: never type a number, status, badge or version by
   hand, and write "missing" for anything it doesn't show. Arithmetic on printed numbers is fine if
   you show the inputs. If a script errors, paste the error and stop. Don't redraw the split or
   change any label. One step per reply: don't run a later section until I send its prompt.
5. Every result reply opens with its answer line, with nothing above it (no file path, check or
   heading). When the step compares the systems, that line is the verdict. Give differences as
   M11 - M10 in 1-5 points, from compare_policies("M11", "M10") (decide() calls it); the summary
   prints M10 - M11, so never relabel its line. Say CJE's checks passed only when the Status line
   reads "good". End each lab reply (Checkpoint 1 to section 5) with these four lines, filled in:
   Labels: SummEval expert ratings, random draws fixed before any run; rater blindness not confirmed.
   Judge: GPT-4, G-Eval consistency prompt, mean of 20 samples, same for both systems.
   cje-eval: <version as printed>
   Files: <script>, <output>
Facts CJE can't check:
- Labels: expert consistency ratings (1-5, the mean of 3 SummEval experts). M10's 30 fit labels and
  the 40 audit documents (labeled on both systems) are fixed random draws, saved before any CJE run.
  M11 has no labels on the 60 evaluation documents. The audit documents are held out: until
  Exercise A′ they enter only through transport=. Whether the experts could see which system wrote
  each summary is not confirmed; say so wherever blind raters matter.
- Judge: one judge scored every row of both systems: GPT-4 with G-Eval's consistency prompt (scores
  released by the G-Eval authors, nlpyang/geval @6f84404). Each score is the mean of 20 samples, 1-5.
- Units: lab/workshop-data.json stores scores and labels as (x - 1) / 4. Section 1's to_points turns
  them back into 1-5 points; report every number in points.
My rule, written before looking: switch to M11 only if CJE marks the result decision-ready and the
paired 95% lower bound of M11 - M10 clears 0 plus any audit tolerance the result relies on; keep
M10 only if the whole interval is below 0.
The audit tolerance for a reused calibration is +/-0.10 points.
Reply with both versions and item 1 of the skill's "Reporting back to the user", quoted word for
word, then wait.
```


## One prompt per checkpoint

**Checkpoint 1 (section 2): look before you fit.** Your guess first: what will CJE say about M11?

```text
Checkpoint 1. Don't fit anything with CJE yet: run the cells through section 2. In 1-5 judge
points, what range do M10's 30 labeled responses cover? How many of M11's 60 evaluation judge scores
fall outside it (count and percent)? CJE marks a policy REFUSE-LEVEL when 5% or more of its judge
scores fall outside the labeled range. From that and the skill's rules, say in two sentences what
CJE will say about M11, and why. Nothing has read the audit labels yet, so don't predict the audit.
Close with the four lines from setup step 5.
```

**Checkpoint 2 (section 3): run the comparison.**

```text
Checkpoint 2. Run the cells through section 4 (stop before Exercise A′), with run() (section 3) and
decide() (setup helpers, section 1) as written: M10's 30 fit labels, M11 borrowing M10's calibration, the 40-document audit at
+/-0.10, scales (1, 5). End the script with print(result.metadata["own_oracle_labels_by_policy"]).
If it doesn't print {'M10': 30, 'M11': 0}, stop and say so. Otherwise open with the verdict in the
skill's borrowed-calibration form ("Undecided: M11 has no labels of its own, ..."), and quote that
print after it. Then give: M11 - M10 with its 95% CI and p-value; each audit's residual, interval
and status; which system's audit matters here (the one that borrows a calibration) and which gates
nothing; each system's support badge; decision_ready; and the action under my rule. Explain these
in one plain sentence each, using these meanings: CAUTION (20% or more of calibrated values sit at
the ends of the map's fitted range), REFUSE-LEVEL (5% or more of judge scores sit outside the
labeled range, so no level claim from this fit), FAIL (the audit interval sits wholly outside the
tolerance), INCONCLUSIVE (it crosses the tolerance, or the audit has fewer than 20 independent
prompt clusters; not a pass) and "Status: critical" (CJE's validity checks failed; not an install
problem). Paste every CJE warning, the summary's Limitations and Status lines, and decision_note
word for word. Don't say which estimate is higher or call either system the leader. Last: judging
from the paired half-width, could this design ever have said switch? Close with the four lines
from setup step 5.
```

**Exercise A′: the same 110 expert labels, used differently.** Write down your prediction first. Will the interval narrow? Will the result be decision-ready? What do we give up?

```text
Exercise A′. Run the cells through the Exercise A′ cell (stop before section 5), as written, and
end the script with print(spent.summary()). All 110 labels become correction labels (M10 70, M11
40, on all 100 documents, no transport=); row 2 still uses the judge, calibrated on each system's
own labels. Row 3 is a paired t-test on the 80 audit labels alone, with no judge. The audit
documents were a random draw, so these are random labels. Verdict line first, for row 2. Then give
all three rows of the cell's table as M11 - M10 in 1-5 points with 95% CIs, decision_ready and the
action under my rule, and note that each row covers a different document set. Paste the Status
line spent.summary() prints. stderr also repeats section 3's warnings (that is row 1), so don't
assign them to row 2; if row 2 prints no warning, say so without guessing why. Quote no M11 level
from the lab design. Then answer each by number: (1) What did we give up compared with
Checkpoint 2? (2) Is M11's FAIL still on the record, and does row 2's estimate rely on transport?
(3) In row 2, what share of each system's 100 documents carries a label? (4) Row 3 is about as wide
as row 2 with 30 fewer labels: why, given (3) and that row 3 is paired on the same 40 documents?
Close with the four lines from setup step 5.
```

**Section 5: size M11's audit.**

```text
Section 5. Run the cells through section 5 (stop before section 6); call plan_transport_audits as
the cell does, never a formula of your own. This step is about M11's audit, the one this decision
relies on. Open with whether any audit size could turn M11's FAIL into a PASS at +/-0.10 (yes or no,
and why in one line). Report each system's spread (expert - map per audit article), and the expert
labels M11's audit would need at +/-0.10, +/-0.20 and +/-0.40, each as a multiple of the 40 we used.
Say what true mean error (AuditScenario's assumed_bias) the cell assumes, and whether that holds for
M11. My +/-0.10 tolerance is fixed; don't propose changing it. Close with the four lines from setup
step 5.
```

**Monday (section 6).** Rehearse on the constructed file first:

```text
Data: lab/toy-evals.csv (constructed: production and candidate on 60 prompts;
a blank human_rating means unlabeled)
Question: is candidate better than production, and by how much?
How the labels were chosen: 30 random responses per policy, at the same rate in every
judge-score bin. The made-up ratings don't depend on which policy wrote the response,
so treat them as blind to policy.
Judge: one made-up judge scored every row on a 1-5 scale. Labels are also 1-5.
Smallest gap that would change the decision, set before looking: +0.10 points,
candidate - production
Read the skill first. Write the script from the skill, loading only lab/toy-evals.csv (not the
notebook's code). Verdict line first. State how the labels were chosen and the judge as facts I
gave you, not as something the data showed. Save the script as agent-runs/monday.py and its
output as agent-runs/monday.txt, with the cje-eval version in both. If a run errors, say so,
and quote only the saved output of a run that finished.
```

Then use the same prompt on your own export:

```text
Install "cje-eval>=0.9.3" in a Python 3.10-3.13 environment. Read `cje skill` and
`cje skill --reference` in full.
Data: [path to your export]
Question: [e.g. is candidate better than production on support prompts, and by how much?]
How the labels were chosen: [e.g. 50 random responses per policy, rated blind to policy /
escalated tickets only / user thumbs / not sure]
Judge: [model + version, rubric version; the same for every row and policy? yes / no / not sure]
Smallest gap that would change the decision, set before looking: [e.g. +0.10 on our 1-5 rubric]
Verdict line first. Save the script, its output and the cje-eval version. If a run errors,
say so, and quote only the saved output of a run that finished.
```

**Extra E3, Exercise B: which bias changes the choice?** Pencil first (3 minutes): fill in every `__` in the prompt, then ask the agent.

```text
Exercise B (constructed numbers, 0-1 scale). Before anything else: if any blank below still reads
__, reply only "Send me your four answers first." and stop. Calibrated judge means: A 0.78, B 0.82.
Mean residual (expert - calibrated): A -0.02, B -0.12. My answers: expert A __, expert B __,
B - A __. If both residuals were -0.10, B - A would be __. Check my arithmetic, then say in three
lines which kind of bias changes the choice and which kind cancels.
```

## Check your agent's reply

- **The verdict line comes first**, before any number or file path: "Undecided: ...", "M10 beats M11", "No reliable winner: ..." or "Blocked: ...". (Checkpoint 1 and section 5 compare nothing, so they open with the step's answer.) At Checkpoint 2 it is "Undecided: M11 has no labels of its own, so the paired difference ... assumes M10's calibration carries over to M11; the true difference can have either sign." If the reply opens with a ranked table or a best guess, stop.
- **Nothing without its own labels is named the winner.** At Checkpoint 2, M11 borrows M10's map, so the reply names no winner and no lean either way. M11 is never "higher", "ahead" or "significant", and M10 is never "the leader".
- **It states how the labels were chosen and which judge scored the rows**, as facts CJE can't check. A GOOD status doesn't cover them. In the lab, these are the four closing lines from setup step 5.
- **It gives both CI bounds in your units** (here, 1-5 points), as candidate - baseline (the summary prints M10 - M11), and compares them with 0 (on your own data, with your margin). A positive M11 - M10 estimate at Checkpoint 2 or Exercise A′ means the sign was flipped.
- **It names its script and output file**, and every number, status and badge it quotes is printed there. To check, pick one and ask which line of the `.txt` prints it.
- **It never says "toss-up", "statistically indistinguishable" or "decide on cost"**, even hedged.

If a check fails, tell the agent which one and ask it to rewrite the reply.

## For you, after each step: checking against the room's outputs

To keep coding agents that read `@AGENT-PATH.md` from seeing the answers ahead of time, the printed tables and numbers live in `workshop-executed.ipynb` (and `workshop-executed.html`), not in this file. Open `workshop-executed.ipynb` (GitHub shows it with every output) after your prediction and agent run to compare your numbers digit for digit:
- Note on Checkpoint 2 (section 3): `result.summary()` prints `M10 - M11` (`+0.556`), while the decision record right below it flips to candidate minus baseline, `M11 - M10` (`-0.556`).
- Checked on Oct 18 with coding agents (Python 3.13, `cje-eval 0.9.3`) on an earlier version of these prompts: frontier-size models matched every number and verdict; a small fast model matched the numbers but typed some results by hand and misexplained Exercise A′. This version of the prompts has not been rerun yet, so use a frontier-size model and the checklist above.

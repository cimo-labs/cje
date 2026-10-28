# Causal ML for AI Evaluation: hands-on lab


ODSC AI West 2026 · Eddie Landesberg (Applied AI, Google; views are my own) · I started CJE, an open-source library, before joining Google · CJE is joint work with Manjari Narayan ([arXiv 2512.11150](https://arxiv.org/abs/2512.11150))

Calibrate LLM-judge scores to a small random sample of expert labels, compare two systems with honest intervals, and audit whether a calibration carries over to a new system. Uses the open-source [`cje-eval`](https://github.com/cimo-labs/cje) package on public SummEval / G-Eval data.

## Before the session (5 minutes)

**Colab:** open **[https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/workshops/odsc-2026/workshop.ipynb](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/workshops/odsc-2026/workshop.ipynb)** and run the first code cell. It installs `cje-eval` and `matplotlib` and downloads the `lab/` data folder, then the notebook runs on its own. Run All once before you come (about 30 seconds the first time).

**Local (Python 3.10-3.13):** this folder is `examples/workshops/odsc-2026` in the [cje repo](https://github.com/cimo-labs/cje/tree/main/examples/workshops/odsc-2026). Get it and work from inside it:

```sh
git clone --depth 1 https://github.com/cimo-labs/cje.git
cd cje/examples/workshops/odsc-2026
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r lab/requirements.txt jupyterlab
jupyter lab workshop.ipynb
```

Run the notebook from this folder: it reads `lab/` and writes `figures/`. The first run also builds matplotlib's font cache, which takes a few silent seconds.

Check `python3 --version` first (`cje-eval==0.9.3` supports Python **3.10–3.13**):
- **Python 3.9 or 3.14:** `pip install -r lab/requirements.txt` stops with `No matching distribution found for cje-eval==0.9.3` (and on 3.9, a bare unpinned `pip install cje-eval` quietly installs the legacy 0.5 line). Create the environment with Python 3.12 (`python3.12 -m venv .venv`, Homebrew `python@3.12`, or `uv venv -p 3.12 .venv`), or use the Colab link above.

**Can't install?** Open `workshop-executed.ipynb` on GitHub (it shows every output), or `workshop-executed.html` in a browser from a local copy, and follow the exercises on paper.

No API keys or model calls. Colab asks for a Google sign-in; a local install runs offline in a few seconds once set up.

**Prefer a coding agent?** [AGENT-PATH.md](AGENT-PATH.md) has a paste-in prompt for each checkpoint and a checklist for the agent's replies (for the room's exact outputs afterward, open `workshop-executed.ipynb`). The notebook stays the main path; the room moves with it.

## What's here

| File | What it is |
|---|---|
| `workshop.ipynb` | The lab. Its first code cell installs what's missing and fetches `lab/` when the notebook is opened on its own |
| `workshop-executed.ipynb`, `workshop-executed.html` | The same notebook, already run |
| `AGENT-PATH.md` | Optional: paste-in prompts to work through the lab with a coding agent instead of the notebook (frontier-size models), plus a checklist for its replies |
| `figures/` | The calibration figure and the three-designs comparison, as PNGs |
| `lab/workshop-data.json` | The numeric extract: 100 documents × 2 systems |
| `lab/artifact-hashes.json` | SHA-256 of every file in `lab/`. The notebook also pins `workshop-data.json`'s hash in its own code |
| `lab/data-check-output.json` | Descriptive checks of the extract |
| `lab/toy-evals.csv`, `lab/make-toy-evals.py` | A **constructed** 120-row example for the "your data" section, and the seeded script that made it |
| `lab/requirements.txt` | `cje-eval==0.9.3`, `matplotlib` |
| `lab/LICENSE-upstream.txt`, `lab/LICENSE-summeval.txt` | G-Eval and SummEval MIT notices |
| `SHA256SUMS.txt` | SHA-256 of every other file in this folder |

## Data and licenses

- Judge scores: GPT-4 outputs released by the G-Eval authors ([nlpyang/geval @6f84404](https://github.com/nlpyang/geval/tree/6f84404ce4503d2ba0cde0293c5d402ecf2976a8)). MIT License, Copyright (c) 2024 Yang Liu.
- Expert consistency ratings (mean of 3 experts): [SummEval](https://github.com/Yale-LILY/SummEval). MIT License, Copyright (c) 2021 Alex Fabbri, Wojciech Kryściński, Bryan McCann, Caiming Xiong, Richard Socher, and Dragomir Radev.
- The extract holds only document and system IDs, numeric judge scores and aggregate ratings. No article or summary text.
- 110 of the 200 ratings are included (30 for fitting, 80 for the audit). The rest are held back so the answer can be checked at the end.

## How the split was drawn

The 60 evaluation and 40 audit documents, and the order in which M10's fit labels are taken, are fixed in `lab/workshop-data.json` and were drawn at random when the extract was built. They come from a seeded script run on Oct 2, 2026: seed 2026 shuffles the 100 documents into 40 audit and 60 evaluation documents, and seed 2027 orders the evaluation documents for M10's fit labels (the 20-label extra takes the first 20 of the same order). The split was saved before the first recorded CJE run on it, and `lab/workshop-data.json` has been byte-identical (sha256 01d66d57…) in every revision since. The two systems were chosen after looking at label distributions, not preregistered: the first pair was replaced because one system's fit labels were constant. The answer key shows this split is unusual: the 30 fit labels ran high, and the M11 audit's FAIL is larger than most random splits give. Read the FAIL's direction as real and its size as luck of the draw.

## Checked on

- macOS arm64 with cje-eval 0.9.2: Python 3.10-3.13 gave the same numbers in the first build check; on Oct 11 this version of the notebook gave identical output on 3.11, 3.12 and 3.13.
- macOS arm64 with cje-eval 0.9.3 (Oct 18; it changes only the bundled agent-skill text): on Python 3.11 the notebook and the facilitator copy gave the same output as on 0.9.2 except the printed version, and the same figures, byte for byte. Re-run end to end on Oct 23 (Python 3.13, cje-eval 0.9.3): same numbers.
- Oct 24, after the clarity pass (Python 3.11.13, cje-eval 0.9.3): every number unchanged except section 5's planner, which now plans M11's audit alone (1,588 / 399 / 102 at ±0.10 / ±0.20 / ±0.40); the decision record lost one row, section 4's follow-up prints only the half-width, and Exercise B moved to the extras (E3). The facilitator copy on Python 3.13 gave the same output except the printed version.
- Opened on its own, with no `lab/` next to it, in a fresh Python 3.12 environment: the first cell installed cje-eval and matplotlib and fetched `lab/` from a local web server standing in for the public link, and every cell ran.
- Without matplotlib (and without pip to install it): the figures are skipped and every number still prints.
- Linux x86_64 (Python 3.13.15, cje-eval 0.9.3, Oct 27): full notebook and standalone setup-cell fetch ran cleanly with identical numbers.
- **Not yet tested: Colab, Windows.**

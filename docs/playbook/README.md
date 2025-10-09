# CJE Practitioners' Playbook

This directory contains the source files for the CJE Practitioners' Playbook, a comprehensive guide for using Causal Judge Evaluation in practice.

## Quick Start

### Build the PDF locally

```bash
cd docs/playbook
make pdf
```

The PDF will be in `build/cje_playbook_vX.Y.Z.pdf` where X.Y.Z is the version from `cje/__version__.py`.

### Build the HTML version

```bash
make html
```

Requires `pandoc` to be installed.

### Build both

```bash
make all
```

### Clean build artifacts

```bash
make clean
```

## Structure

```
docs/playbook/
├── main.tex              # Main LaTeX document
├── sections/             # Individual sections
│   ├── 01_introduction.tex
│   ├── 02_dm.tex
│   ├── 03_offpolicy.tex
│   ├── 04_diagnostics.tex
│   ├── 05_assumptions.tex
│   ├── 06_playbook.tex
│   ├── 07_cases.tex
│   ├── 08_implementation.tex
│   ├── 09_limitations.tex
│   └── 10_conclusion.tex
├── figs/                 # Figures and diagrams
├── style/                # LaTeX style files
│   └── cje.sty          # Custom CJE styling
├── references.bib        # Bibliography
├── Makefile             # Build automation
└── build/               # Output directory (gitignored)
```

## Version Management

The playbook version is **automatically extracted** from `cje/__version__.py`:

1. When you run `make pdf`, it generates `VERSION.tex` with the current package version
2. This version appears in:
   - The document header/footer
   - The title page
   - The output filename

This ensures the playbook is always synchronized with the code version.

## CI/CD

The playbook is automatically built by GitHub Actions:

- **On every push to `main`**: Builds a development version
- **On version tags** (e.g., `v0.2.1`): Builds and attaches PDF/HTML to the GitHub Release

See `.github/workflows/build_playbook.yml` for details.

## Writing Content

### Adding content to sections

Edit the appropriate file in `sections/`. For example:

```bash
vim sections/02_dm.tex
```

### Adding figures

1. Place figure files (PDF, PNG) in `figs/`
2. Reference in LaTeX:

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\textwidth]{figs/standard_panel.pdf}
  \caption{CJE Standard Panel}
  \label{fig:standard-panel}
\end{figure}
```

### Using custom commands

The `style/cje.sty` package provides shortcuts:

- `\cje` → renders as formatted "CJE"
- `\dm`, `\ips`, `\dr` → formatted estimator names
- `\autocal`, `\simcal`, `\oua` → formatted component names
- `\E`, `\Var`, `\SE` → math operators
- `\est{V}` → produces V-hat

Example:

```latex
The \dm estimator computes $\est{V}(\pi) = \E[R]$ using \autocal.
```

### Using custom environments

```latex
\begin{quickref}
Quick Reference:
- Input: X, Y, Z
- Output: Estimate with CI
\end{quickref}

\begin{recipe}
1. Step one
2. Step two
3. Step three
\end{recipe}
```

## Requirements

### For PDF building

- LaTeX distribution (TeXLive or MiKTeX)
- `pdflatex`
- `bibtex`

On Ubuntu/Debian:
```bash
sudo apt-get install texlive-full
```

On macOS with Homebrew:
```bash
brew install --cask mactex
```

### For HTML building (optional)

- `pandoc`

```bash
# Ubuntu/Debian
sudo apt-get install pandoc

# macOS
brew install pandoc
```

## Troubleshooting

### "Command not found: pdflatex"

Install a LaTeX distribution (see Requirements above).

### "Version extraction failed"

Ensure you're running make from the `docs/playbook/` directory and that `cje/__version__.py` exists at `../../cje/__version__.py`.

Test with:
```bash
make test-version
```

### "Bibliography not found"

Run the full build sequence:
```bash
make clean
make pdf
```

This runs pdflatex → bibtex → pdflatex → pdflatex.

## Publishing

When you tag a release:

```bash
git tag v0.3.0
git push origin v0.3.0
```

GitHub Actions will:
1. Build `cje_playbook_v0.3.0.pdf`
2. Build `cje_playbook_v0.3.0.html`
3. Attach both to the v0.3.0 GitHub Release

You can then link to these from your website or documentation.

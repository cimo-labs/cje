# Concatenated Playbook

## Purpose

`FULL_PLAYBOOK_CONCAT.tex` combines all playbook .tex files into a single file for easy sharing with AI assistants or reviewers.

## Regenerate

```bash
cd docs/playbook
./concat_playbook.sh
```

This creates `FULL_PLAYBOOK_CONCAT.tex` (~2200 lines) with all sections in order:
1. main.tex (preamble, document setup)
2. sections/01_introduction.tex
3. sections/02_dm.tex
4. sections/03_offpolicy.tex
5. sections/04_diagnostics.tex
6. sections/05_assumptions.tex
7. sections/06_playbook.tex
8. sections/07_cases.tex
9. sections/08_implementation.tex
10. sections/09_limitations.tex
11. sections/10_conclusion.tex

Each section is clearly marked with `%%%%%%%% sections/XX.tex %%%%%%%%` headers.

## Note

`FULL_PLAYBOOK_CONCAT.tex` is in `.gitignore` as it's a generated file. Regenerate it whenever the playbook changes.

# CJE Playbook - Web Build System

This directory contains the CJE Practitioners' Playbook LaTeX source and web build system.

## Quick Reference

**Edit content:** `sections/*.tex`
**Build HTML:** `make web` (requires pandoc)
**Build PDF:** `make pdf` (requires pdflatex)
**Auto-build:** GitHub Actions builds and commits HTML on push to main

## Architecture

### Single Source of Truth

LaTeX files in `sections/` are the source of truth. From these, we generate:
- **PDF** - Full playbook document (via pdflatex)
- **Web HTML** - Individual section pages (via pandoc)

### Build System

**Local builds:**
```bash
make pdf   # Requires: pdflatex, bibtex
make web   # Requires: pandoc
```

**CI builds (automatic):**
- Triggered on push to main branch or tags
- Builds PDF and web HTML using GitHub Actions
- **Commits rendered HTML back to repo** (build/web/)
- No local tools needed for website sync!

### Web HTML Output

HTML files are built to `build/web/`:
- `*.html` - Section content (one per .tex file)
- `manifest.json` - Navigation metadata
- `figs/` - Figures (if any)

These files are **committed to git** (exception in .gitignore) so the website can sync without building.

## Workflow

### For playbook content changes:

1. **Edit** LaTeX in `sections/*.tex`
2. **Commit and push** to main branch
3. **Wait** ~2 minutes for GitHub Actions
4. **Pull** to get the auto-committed HTML
5. **Sync to website** (in website repo: `npm run sync-playbook`)

### For local testing:

```bash
# Build web HTML locally
make web

# View output
ls -lh build/web/
```

## GitHub Actions

Workflow file: `.github/workflows/build_playbook.yml`

**Triggers:**
- Push to main branch
- Version tags (v*)
- PRs affecting playbook files

**Actions:**
1. Build PDF and upload as artifact
2. Build web HTML with pandoc
3. **Commit HTML to build/web/** (main branch only)
4. Attach PDF and HTML to releases (on tags)

**Auto-commit:** The workflow commits rendered HTML with message "Auto-update playbook web HTML [skip ci]"

## Files

```
docs/playbook/
├── sections/           # LaTeX source files
│   ├── 01_introduction.tex
│   ├── 02_dm.tex
│   └── ...
├── build/              # Generated files (gitignored except web/)
│   ├── *.pdf          # PDF output
│   └── web/           # Web HTML (committed to git!)
│       ├── *.html
│       └── manifest.json
├── style/             # LaTeX style files
├── Makefile           # Build commands
├── build_web.sh       # Web HTML converter (uses pandoc)
└── main.tex           # LaTeX master file
```

## Pandoc Conversion

`build_web.sh` converts each `.tex` file to HTML fragments using:
- `--from=latex` - Parse LaTeX syntax
- `--to=html` - Output HTML fragments (NOT standalone)
- `--mathjax` - Preserve MathJax-compatible math markup
- `--section-divs` - Wrap sections in divs
- `--no-highlight` - Website provides code styling

Note: We generate **content fragments**, not complete HTML documents. The website provides the HTML wrapper, CSS styling, and loads MathJax.

## Dependencies

**For PDF:**
- pdflatex
- bibtex

**For Web HTML:**
- pandoc (installed in CI, optional locally)
- python3 (for version extraction)

**For website sync:**
- Nothing! Just copy pre-rendered files.

## Adding a Section

1. Create `sections/##_newsection.tex`
2. Update section list in `build_web.sh` (around line 67)
3. Update website's `src/lib/playbook.ts` if needed
4. Push to main → CI builds automatically

## Troubleshooting

**"pandoc: command not found"**
- Install: `brew install pandoc` (macOS) or `sudo apt-get install pandoc` (Linux)
- Or wait for GitHub Actions to build

**HTML not in git**
- Check `.gitignore` has exceptions for `!docs/playbook/build/web/`
- Ensure GitHub Actions workflow completed successfully

**Math not rendering**
- Website needs MathJax loaded (already configured in layout.tsx)
- Check browser console for errors

## Website Integration

The CIMO Labs website syncs playbook content from this repo:
- **Location:** `/Users/eddielandesberg/CJE/cje` → website `public/playbook/`
- **Command:** `npm run sync-playbook` (in website repo)
- **Process:** Copies `build/web/` contents (no build needed)

See website repo's `PLAYBOOK_WEB.md` for details.

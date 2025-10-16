#!/bin/bash
# Build web-friendly HTML from LaTeX playbook using pandoc
# Designed to run in CI (pandoc available) and commit the output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SECTIONS_DIR="$SCRIPT_DIR/sections"
WEB_DIR="$BUILD_DIR/web"

echo "🔨 Building web version of CJE Playbook..."

# Get version
VERSION="0.0.0"
if [ -f "$SCRIPT_DIR/../../cje/__init__.py" ]; then
    VERSION=$(python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/../..'); from cje import __version__; print(__version__)" 2>/dev/null || echo "0.0.0")
fi

echo "Version: $VERSION"

# Create output directory
mkdir -p "$WEB_DIR"

# Check if pandoc is available
if ! command -v pandoc &> /dev/null; then
    echo "❌ Error: pandoc is required but not installed."
    echo "   Install: sudo apt-get install pandoc  (Linux)"
    echo "   Install: brew install pandoc          (macOS)"
    exit 1
fi

# Pandoc options for clean HTML output
# Note: NOT using --standalone so we get pure content fragments
# The website provides its own HTML wrapper, CSS, and MathJax
PANDOC_OPTS=(
    "--from=latex"
    "--to=html"
    "--mathjax"  # Preserve MathJax-compatible markup (website loads MathJax)
    "--section-divs"  # Wrap sections in divs
    "--no-highlight"  # We'll handle code highlighting in CSS
)

# Convert each section
echo "Converting sections..."
for tex_file in "$SECTIONS_DIR"/*.tex; do
    if [ -f "$tex_file" ]; then
        filename=$(basename "$tex_file" .tex)
        echo "  → $filename"

        # Run pandoc
        pandoc "${PANDOC_OPTS[@]}" \
            "$tex_file" \
            -o "$WEB_DIR/$filename.html"
    fi
done

# Generate manifest.json
echo "📝 Generating manifest..."

cat > "$WEB_DIR/manifest.json" <<EOF
{
  "version": "$VERSION",
  "generated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "sections": [
    {"id": "01_introduction", "title": "Introduction", "slug": "introduction"},
    {"id": "02_dm", "title": "Direct Method (DM)", "slug": "direct-method"},
    {"id": "03_offpolicy", "title": "Off-Policy Evaluation", "slug": "off-policy"},
    {"id": "04_diagnostics", "title": "Diagnostics & Fixes", "slug": "diagnostics"},
    {"id": "05_assumptions", "title": "Assumptions", "slug": "assumptions"},
    {"id": "06_playbook", "title": "Operator Playbook", "slug": "playbook"},
    {"id": "07_cases", "title": "Case Studies", "slug": "case-studies"},
    {"id": "08_implementation", "title": "Implementation Guide", "slug": "implementation"},
    {"id": "09_limitations", "title": "Limitations", "slug": "limitations"},
    {"id": "10_conclusion", "title": "Conclusion", "slug": "conclusion"}
  ]
}
EOF

# Copy images if they exist
echo "📦 Copying images..."
if [ -d "$SCRIPT_DIR/figs" ]; then
    mkdir -p "$WEB_DIR/figs"
    cp -r "$SCRIPT_DIR/figs/"* "$WEB_DIR/figs/" 2>/dev/null || true
fi

for img in "$SCRIPT_DIR"/*.png "$SCRIPT_DIR"/*.jpg "$SCRIPT_DIR"/*.svg; do
    if [ -f "$img" ]; then
        cp "$img" "$WEB_DIR/"
    fi
done

echo "✅ Web build complete: $WEB_DIR"
echo "   Sections: $(ls "$WEB_DIR"/*.html 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "   These files should be committed to git for easy sync to website."

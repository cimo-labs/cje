# CJE Playbook Makefile
# Auto-builds PDF and HTML with version injection from cje/__version__.py

# Paths
PYTHON := python3
CJE_ROOT := ../..
VERSION_FILE := VERSION.tex
MAIN_TEX := main.tex
MAIN_PDF := main.pdf
BUILD_DIR := build

# Extract version from Python package
get_version:
	@$(PYTHON) -c "import sys; sys.path.insert(0, '$(CJE_ROOT)'); from cje import __version__; print(__version__)"

# Generate VERSION.tex file
$(VERSION_FILE):
	@echo "Extracting version from cje package..."
	@$(PYTHON) -c "import sys; sys.path.insert(0, '$(CJE_ROOT)'); from cje import __version__; print('\\\\newcommand{\\\\cjeversion}{' + __version__ + '}')" > $@
	@cat $@

# Build PDF
pdf: $(VERSION_FILE)
	@echo "Building PDF with version..."
	pdflatex $(MAIN_TEX)
	bibtex main || true
	pdflatex $(MAIN_TEX)
	pdflatex $(MAIN_TEX)
	@mkdir -p $(BUILD_DIR)
	@VERSION=$$($(PYTHON) -c 'import sys; sys.path.insert(0, "$(CJE_ROOT)"); from cje import __version__; print(__version__)'); \
	mv $(MAIN_PDF) $(BUILD_DIR)/cje_playbook_v$$VERSION.pdf
	@echo "Built: $(BUILD_DIR)/cje_playbook_v$$($(PYTHON) -c 'import sys; sys.path.insert(0, \"$(CJE_ROOT)\"); from cje import __version__; print(__version__)').pdf"

# Build HTML (requires pandoc)
html: $(VERSION_FILE)
	@echo "Building HTML with version..."
	@mkdir -p $(BUILD_DIR)
	@VERSION=$$($(PYTHON) -c 'import sys; sys.path.insert(0, "$(CJE_ROOT)"); from cje import __version__; print(__version__)'); \
	pandoc $(MAIN_TEX) -s --bibliography=references.bib --citeproc --mathml \
		-o $(BUILD_DIR)/cje_playbook_v$$VERSION.html || echo "pandoc not installed, skipping HTML build"
	@echo "Built: $(BUILD_DIR)/cje_playbook_v$$($(PYTHON) -c 'import sys; sys.path.insert(0, \"$(CJE_ROOT)\"); from cje import __version__; print(__version__)').html"

# Build both PDF and HTML
all: pdf html

# Clean build artifacts
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
	rm -f *.fls *.fdb_latexmk *.synctex.gz
	rm -f $(VERSION_FILE)
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build artifacts"

# Clean and rebuild
rebuild: clean all

# Test version extraction
test-version:
	@echo "Testing version extraction..."
	@$(PYTHON) -c "import sys; sys.path.insert(0, '$(CJE_ROOT)'); from cje import __version__; print('CJE version:', __version__)"

.PHONY: get_version pdf html all clean rebuild test-version

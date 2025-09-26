# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Fast build (skip API docs for development)
fasthtml:
	$(SPHINXBUILD) -E -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) -D autodoc_mock_imports="numpy,pandas,torch,sklearn"

# Production build with optimizations
prodhtml:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) -j auto -T -W --keep-going

# Build and serve locally
serve: html
	@echo "🌐 Starting local server..."
	cd $(BUILDDIR)/html && python -m http.server 8000

# Clean build with cache clearing
cleanall: clean
	rm -rf $(BUILDDIR)
	rm -rf api/generated/
	rm -rf _generated/

.PHONY: fasthtml prodhtml serve cleanall

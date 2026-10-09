# CJE Makefile

.PHONY: test test-examples lint format sync-skill help

# Documentation now hosted on cimolabs.com

# Development commands
test:  ## Run tests
	poetry run pytest cje/tests/ -v

test-examples:  ## Run documented workflows and research examples (poetry install --with research)
	poetry run pytest cje/tests/test_examples.py cje/tests/test_doc_workflows.py experiments -v

lint:  ## Run linting
	poetry run black --check cje/
	poetry run mypy cje/ --ignore-missing-imports

format:  ## Format code
	poetry run black cje/

sync-skill:  ## Copy skills/cje/{SKILL.md,reference.md} into the bundled cje/.agents/skills/cje/
	mkdir -p cje/.agents/skills/cje
	cp skills/cje/SKILL.md skills/cje/reference.md cje/.agents/skills/cje/

# Installation
install:  ## Install package
	poetry install

dev-setup:  ## Set up development environment
	poetry install
	pre-commit install

# Help
help:  ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

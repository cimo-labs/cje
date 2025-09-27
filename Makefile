# CJE Makefile

.PHONY: test lint format help

# Documentation now hosted on cimo-labs.com

# Development commands
test:  ## Run tests
	poetry run pytest cje/tests/ -v

lint:  ## Run linting
	poetry run black cje/
	poetry run mypy cje/ --ignore-missing-imports

format:  ## Format code
	poetry run black cje/

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
.PHONY: help install install-dev lint format check test clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:  ## Run linting checks
	ruff check .
	mypy .

format:  ## Format code
	ruff format .
	black .
	isort .

check: lint  ## Run all checks (lint + format check)
	ruff format --check .
	black --check .
	isort --check-only .

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=. --cov-report=html --cov-report=term

clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/ dist/ build/

fix:  ## Fix code issues automatically
	ruff check --fix .
	ruff format .
	black .
	isort .

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files
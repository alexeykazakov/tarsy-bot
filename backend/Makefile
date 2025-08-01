# Tarsy Backend - Makefile
# ==========================

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
UV := uv
TEST_DIR := tests
APP_DIR := app

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(GREEN)Tarsy Backend - Available Commands$(NC)"
	@echo "==================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Setup and Installation
.PHONY: install
install: ## Install all dependencies (production + development + test)
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(UV) sync
	$(UV) sync --extra dev --extra test

.PHONY: install-prod
install-prod: ## Install production dependencies only
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(UV) sync

.PHONY: install-dev
install-dev: ## Install development dependencies (includes test)
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(UV) sync --extra dev --extra test

# Testing
.PHONY: test
test: install ## Run all tests (unit + integration)
	@echo "$(GREEN)Running all tests...$(NC)"
	TESTING=true . .venv/bin/activate && $(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-unit
test-unit: install ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	TESTING=true . .venv/bin/activate && $(PYTHON) -m pytest tests/unit/ -v --tb=short

.PHONY: test-integration
test-integration: install ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	TESTING=true . .venv/bin/activate && $(PYTHON) -m pytest tests/integration/ -v --tb=short

.PHONY: test-verbose
test-verbose: ## Run all tests with verbose output
	@echo "$(GREEN)Running all tests (verbose)...$(NC)"
	TESTING=true . .venv/bin/activate && $(PYTHON) -m pytest tests/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	TESTING=true . .venv/bin/activate && $(PYTHON) -m pytest tests/ --cov=tarsy --cov-report=html --cov-report=term-missing

.PHONY: test-quick
test-quick: ## Run tests in parallel for faster execution
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	TESTING=true . .venv/bin/activate && $(PYTHON) -m pytest tests/ -n auto

# Code Quality
.PHONY: lint
lint: ## Run linting checks with ruff
	@echo "$(GREEN)Running linting checks...$(NC)"
	$(UV) run ruff check $(APP_DIR) $(TEST_DIR)

.PHONY: lint-fix
lint-fix: ## Run linting checks and auto-fix issues
	@echo "$(GREEN)Running linting with auto-fix...$(NC)"
	$(UV) run ruff check --fix $(APP_DIR) $(TEST_DIR)

.PHONY: format
format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	$(UV) run black $(APP_DIR) $(TEST_DIR)

.PHONY: format-check
format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(UV) run black --check $(APP_DIR) $(TEST_DIR)

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(NC)"
	$(UV) run mypy $(APP_DIR)

.PHONY: quality
quality: lint format-check type-check ## Run all code quality checks
	@echo "$(GREEN)All code quality checks completed!$(NC)"

.PHONY: fix
fix: lint-fix format ## Auto-fix linting issues and format code
	@echo "$(GREEN)Code fixes and formatting applied!$(NC)"

# Development Server
.PHONY: dev
dev: ## Start development server with auto-reload
	@echo "$(GREEN)Starting development server...$(NC)"
	. .venv/bin/activate && uvicorn tarsy.main:app --reload --port 8000 --log-level info

.PHONY: dev-debug
dev-debug: ## Start development server with debug logging
	@echo "$(GREEN)Starting development server (debug mode)...$(NC)"
	. .venv/bin/activate && uvicorn tarsy.main:app --reload --port 8000 --log-level debug

.PHONY: run
run: ## Start development server with manual venv activation
	@echo "$(GREEN)Starting development server (manual venv)...$(NC)"
	. .venv/bin/activate && uvicorn tarsy.main:app --reload --port 8000

.PHONY: stop
stop: ## Stop backend server running on port 8000
	@echo "$(GREEN)Stopping backend server...$(NC)"
	@lsof -ti:8000 | xargs -r kill -9 || echo "No process found on port 8000"
	@echo "$(GREEN)Backend server stopped$(NC)"

# Database and Cleanup
.PHONY: clean
clean: ## Clean up generated files and caches
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf .ruff_cache/ 2>/dev/null || true

.PHONY: clean-db
clean-db: ## Clean up database files (USE WITH CAUTION)
	@echo "$(RED)WARNING: This will delete database files!$(NC)"
	@printf "Are you sure? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[Yy]|[Yy][Ee][Ss]) \
			echo "$(YELLOW)Removing database files...$(NC)"; \
			rm -f history.db history.db-shm history.db-wal; \
			echo "$(GREEN)Database files removed$(NC)"; \
			;; \
		*) \
			echo "$(GREEN)Cancelled$(NC)"; \
			;; \
	esac

# Utility targets
.PHONY: check
check: quality test-unit ## Run quality checks and unit tests (fast CI check)
	@echo "$(GREEN)Quick check completed successfully!$(NC)"

.PHONY: ci
ci: quality test ## Run full CI pipeline (quality + all tests)
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

.PHONY: logs
logs: ## Show recent application logs
	@echo "$(GREEN)Recent application logs:$(NC)"
	@if [ -f logs/tarsy.log ]; then tail -n 50 logs/tarsy.log; else echo "No application logs found"; fi

.PHONY: logs-llm
logs-llm: ## Show recent LLM communication logs
	@echo "$(GREEN)Recent LLM communication logs:$(NC)"
	@if [ -f logs/llm_communications.log ]; then tail -n 50 logs/llm_communications.log; else echo "No LLM logs found"; fi

.PHONY: status
status: ## Show project status and health
	@echo "$(GREEN)Tarsy Backend Status$(NC)"
	@echo "===================="
	@echo "Python version: $$(python3 --version)"
	@echo "UV version: $$(uv --version)"
	@echo "Dependencies: $$(if [ -f uv.lock ]; then echo 'Locked'; else echo 'Not locked'; fi)"
	@echo "Tests: $$(find tests/ -name 'test_*.py' | wc -l) test files"
	@echo "Database: $$(if [ -f history.db ]; then echo 'Present'; else echo 'Not found'; fi)"
	@echo "Logs: $$(if [ -d logs ]; then ls logs/ | wc -l; else echo '0'; fi) log files"

# Export variables for sub-processes
export PYTHONPATH := $(shell pwd):$(PYTHONPATH) 
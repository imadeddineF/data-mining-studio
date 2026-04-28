.PHONY: help install run clean venv lint format check freeze dev

PYTHON ?= python3
VENV   ?= .venv
PIP     = $(VENV)/bin/pip
PY      = $(VENV)/bin/python
STREAMLIT = $(VENV)/bin/streamlit
APP     = app.py
PORT   ?= 8501

help: ## Show this help
	@echo ""
	@echo "  Data Mining Studio — available targets"
	@echo "  ---------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""

venv: ## Create the Python virtualenv
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip >/dev/null
	@echo "✓ virtualenv ready at $(VENV)"

install: venv ## Install all dependencies
	@$(PIP) install -r requirements.txt
	@echo "✓ dependencies installed"

run: install ## Launch the Streamlit app
	@$(STREAMLIT) run $(APP) --server.port=$(PORT)

dev: install ## Launch in development mode (auto-reload)
	@$(STREAMLIT) run $(APP) --server.port=$(PORT) --server.runOnSave=true

freeze: ## Lock current dependency versions
	@$(PIP) freeze > requirements.lock.txt
	@echo "✓ wrote requirements.lock.txt"

lint: ## Quick syntax check
	@$(PY) -m compileall -q src app.py && echo "✓ syntax OK"

clean: ## Remove caches and build artifacts
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .streamlit/cache build dist *.egg-info
	@echo "✓ cleaned"

clean-all: clean ## Also remove the virtualenv
	@rm -rf $(VENV)
	@echo "✓ virtualenv removed"

.DEFAULT_GOAL := help

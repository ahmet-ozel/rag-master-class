.PHONY: help setup demo-classical demo-agentic evaluate test clean

.DEFAULT_GOAL := help

help: ## Show available commands
	@echo "RAG Portfolio - Available Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-18s %s\n", $$1, $$2}'
	@echo ""

setup: ## Install Python dependencies and start Docker services
	pip install -r Classical-RAG/requirements.txt
	pip install -r Agentic-RAG/requirements.txt
	@if [ -f evaluation/requirements.txt ]; then pip install -r evaluation/requirements.txt; fi
	docker compose up -d

demo-classical: ## Run Classical RAG demo pipeline
	python Classical-RAG/demo_pipeline.py

demo-agentic: ## Run Agentic RAG demo pipeline
	python Agentic-RAG/agent_demo.py

evaluate: ## Run RAG evaluation script
	python evaluation/evaluate.py

test: ## Run pytest tests
	pytest tests/ -v

clean: ## Remove temporary files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf chroma_db/
	@echo "Cleaned up temporary files."

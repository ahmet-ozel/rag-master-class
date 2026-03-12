.PHONY: help setup demo-classical demo-agentic evaluate test clean

.DEFAULT_GOAL := help

help: ## Show available commands
	@echo "RAG Master Class - Available Commands"
	@echo "======================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-18s %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables (set in .env):"
	@echo "  LLM_PROVIDER    openai | gemini | claude | ollama | vllm"
	@echo "  VECTOR_STORE    chroma | faiss"
	@echo ""

setup: ## Install Python dependencies and start Docker services
	pip install -r Classical-RAG/requirements.txt
	pip install -r Agentic-RAG/requirements.txt
	@if [ -f evaluation/requirements.txt ]; then pip install -r evaluation/requirements.txt; fi
	docker compose up -d

demo-classical: ## Run Classical RAG demo (use LLM_PROVIDER and VECTOR_STORE env vars)
	python Classical-RAG/demo_pipeline.py

demo-agentic: ## Run Agentic RAG demo (use LLM_PROVIDER env var)
	python Agentic-RAG/agent_demo.py

evaluate: ## Run RAG evaluation script
	python evaluation/evaluate.py

test: ## Run pytest tests
	pytest tests/ -v

clean: ## Remove temporary files, caches, and vector stores
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache chroma_db/ faiss_index/
	@echo "Cleaned up."

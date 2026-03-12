"""
Classical RAG Demo Pipeline
============================
Mevcut chunking.py modülü üzerine inşa edilen uçtan uca RAG pipeline.
Çoklu LLM provider (OpenAI, Gemini, Claude, Ollama, vLLM) ve
çoklu vector store (ChromaDB, FAISS) desteği sunar.

Kullanım:
    python demo_pipeline.py --provider openai --data-dir ../examples/data
    python demo_pipeline.py --provider gemini --model gemini-2.0-flash
    python demo_pipeline.py --provider claude --model claude-sonnet-4-20250514
    python demo_pipeline.py --provider ollama --model llama3.2
    python demo_pipeline.py --provider vllm --model meta-llama/Llama-3.2-3B-Instruct
    python demo_pipeline.py --vector-store faiss --provider openai

Genişletilebilirlik Notları:
    - OCR: PaddleOCR, Tesseract veya bulut Vision modelleri entegre edilebilir.
    - Reranking: bge-reranker-v2-m3 veya Cohere Rerank 3.5 eklenebilir.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors."""


class LLMConfigError(RAGPipelineError):
    """LLM configuration is invalid."""


class LLMConnectionError(RAGPipelineError):
    """LLM service is unreachable."""


class VectorStoreError(RAGPipelineError):
    """Vector store connection or operation failed."""


logger = logging.getLogger("demo_pipeline")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    answer: str
    sources: List[ChunkResult]
    model_used: str
    provider: str


# ---------------------------------------------------------------------------
# LLM Provider Protocol & Implementations
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Sen yardımcı bir asistansın. Sana verilen bağlam bilgisini kullanarak "
    "soruları yanıtla. Bağlamda bilgi yoksa, bilmediğini belirt."
)


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str, context: str) -> str: ...


class OpenAIProvider:
    """OpenAI API (GPT-4o, GPT-4o-mini, vb.)"""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key or resolved_key.startswith("your-"):
            raise LLMConfigError(
                "OPENAI_API_KEY bulunamadı.\n"
                ".env dosyanızda OPENAI_API_KEY değerini ayarlayın."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError("openai paketi yüklü değil: pip install openai") from exc

        self._client = OpenAI(api_key=resolved_key)
        self.model = model
        self.provider_name = "openai"

    def generate(self, prompt: str, context: str) -> str:
        user_msg = f"Bağlam:\n{context}\n\nSoru: {prompt}"
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            err = str(exc).lower()
            if "api key" in err or "401" in err:
                raise LLMConfigError("OpenAI API anahtarı geçersiz.") from exc
            raise LLMConnectionError(f"OpenAI hatası: {exc}") from exc


class GeminiProvider:
    """Google Gemini API"""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash") -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key or resolved_key.startswith("your-"):
            raise LLMConfigError(
                "GEMINI_API_KEY bulunamadı.\n"
                ".env dosyanızda GEMINI_API_KEY değerini ayarlayın."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise LLMConfigError("google-genai paketi yüklü değil: pip install google-genai") from exc

        self._client = genai.Client(api_key=resolved_key)
        self.model = model
        self.provider_name = "gemini"

    def generate(self, prompt: str, context: str) -> str:
        user_msg = f"Sistem: {_SYSTEM_PROMPT}\n\nBağlam:\n{context}\n\nSoru: {prompt}"
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user_msg,
            )
            return resp.text or ""
        except Exception as exc:
            err = str(exc).lower()
            if "api key" in err or "403" in err or "401" in err:
                raise LLMConfigError("Gemini API anahtarı geçersiz.") from exc
            raise LLMConnectionError(f"Gemini hatası: {exc}") from exc


class ClaudeProvider:
    """Anthropic Claude API"""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514") -> None:
        resolved_key = api_key or os.getenv("CLAUDE_API_KEY", "")
        if not resolved_key or resolved_key.startswith("your-"):
            raise LLMConfigError(
                "CLAUDE_API_KEY bulunamadı.\n"
                ".env dosyanızda CLAUDE_API_KEY değerini ayarlayın."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigError("anthropic paketi yüklü değil: pip install anthropic") from exc

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.provider_name = "claude"

    def generate(self, prompt: str, context: str) -> str:
        user_msg = f"Bağlam:\n{context}\n\nSoru: {prompt}"
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text if resp.content else ""
        except Exception as exc:
            err = str(exc).lower()
            if "api key" in err or "401" in err:
                raise LLMConfigError("Claude API anahtarı geçersiz.") from exc
            raise LLMConnectionError(f"Claude hatası: {exc}") from exc


class OllamaProvider:
    """Ollama local model"""

    def __init__(self, base_url: str = "", model: str = "llama3.2") -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.provider_name = "ollama"
        self._check_connection()

    def _check_connection(self) -> None:
        import urllib.request, urllib.error
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, OSError) as exc:
            raise LLMConnectionError(
                f"Ollama servisine bağlanılamadı ({self.base_url}).\n"
                "Çalıştırın: ollama serve"
            ) from exc

    def generate(self, prompt: str, context: str) -> str:
        try:
            import ollama as ollama_lib
        except ImportError as exc:
            raise LLMConfigError("ollama paketi yüklü değil: pip install ollama") from exc

        user_msg = f"Bağlam:\n{context}\n\nSoru: {prompt}"
        try:
            client = ollama_lib.Client(host=self.base_url)
            resp = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp["message"]["content"]
        except Exception as exc:
            err = str(exc).lower()
            if "not found" in err or "pull" in err:
                raise LLMConfigError(f"Model '{self.model}' bulunamadı: ollama pull {self.model}") from exc
            raise LLMConnectionError(f"Ollama hatası: {exc}") from exc


class VLLMProvider:
    """vLLM self-hosted model (OpenAI-compatible API)"""

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
    ) -> None:
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        self.provider_name = "vllm"
        resolved_key = api_key or os.getenv("VLLM_API_KEY", "dummy")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError("openai paketi yüklü değil: pip install openai") from exc

        self._client = OpenAI(base_url=f"{self.base_url}/v1", api_key=resolved_key)

    def generate(self, prompt: str, context: str) -> str:
        user_msg = f"Bağlam:\n{context}\n\nSoru: {prompt}"
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            raise LLMConnectionError(
                f"vLLM hatası ({self.base_url}): {exc}\n"
                "vLLM sunucusunun çalıştığından emin olun."
            ) from exc


# ---------------------------------------------------------------------------
# Vector Store Abstraction
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    def upsert(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]) -> None: ...
    def query(self, query_embedding: List[float], top_k: int) -> List[ChunkResult]: ...
    def count(self) -> int: ...


class ChromaVectorStore:
    """ChromaDB vector store."""

    def __init__(self, persist_dir: str = "", collection_name: str = "rag_demo") -> None:
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError("chromadb yüklü değil: pip install chromadb") from exc

        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB bağlantı hatası ({persist_dir}): {exc}") from exc

    def upsert(self, ids, embeddings, documents, metadatas):
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: List[float], top_k: int) -> List[ChunkResult]:
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                chunks.append(ChunkResult(text=doc, score=1.0 - dist, metadata=meta))
        return chunks

    def count(self) -> int:
        return self._collection.count()


class FAISSVectorStore:
    """FAISS vector store with metadata persistence."""

    def __init__(self, index_dir: str = "", collection_name: str = "rag_demo") -> None:
        self._index_dir = index_dir or os.getenv("FAISS_INDEX_DIR", "./faiss_index")
        self._collection_name = collection_name
        os.makedirs(self._index_dir, exist_ok=True)

        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("faiss yüklü değil: pip install faiss-cpu") from exc

        self._faiss = faiss
        self._index: Optional[Any] = None
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []
        self._load()

    def _index_path(self) -> str:
        return os.path.join(self._index_dir, f"{self._collection_name}.index")

    def _meta_path(self) -> str:
        return os.path.join(self._index_dir, f"{self._collection_name}.meta.npz")

    def _load(self) -> None:
        import json
        idx_path = self._index_path()
        meta_path = self._meta_path()
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self._index = self._faiss.read_index(idx_path)
            data = np.load(meta_path, allow_pickle=True)
            self._documents = data["documents"].tolist()
            self._metadatas = [json.loads(m) for m in data["metadatas"].tolist()]
            self._ids = data["ids"].tolist()

    def _save(self) -> None:
        import json
        if self._index is not None:
            self._faiss.write_index(self._index, self._index_path())
            np.savez(
                self._meta_path(),
                documents=np.array(self._documents, dtype=object),
                metadatas=np.array([json.dumps(m) for m in self._metadatas], dtype=object),
                ids=np.array(self._ids, dtype=object),
            )

    def upsert(self, ids, embeddings, documents, metadatas):
        vectors = np.array(embeddings, dtype=np.float32)
        dim = vectors.shape[1]

        if self._index is None:
            self._index = self._faiss.IndexFlatIP(dim)  # inner product (cosine with normalized vectors)

        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Remove existing ids (simple upsert)
        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._ids.pop(idx)
                self._documents.pop(idx)
                self._metadatas.pop(idx)

        self._index.add(vectors)
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._save()

    def query(self, query_embedding: List[float], top_k: int) -> List[ChunkResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            chunks.append(ChunkResult(
                text=self._documents[idx],
                score=float(score),
                metadata=self._metadatas[idx],
            ))
        return chunks

    def count(self) -> int:
        return self._index.ntotal if self._index else 0


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (English, Fast)",
    "all-MiniLM-L12-v2": "all-MiniLM-L12-v2 (English, Balanced)",
    "paraphrase-multilingual-mpnet-base-v2": "paraphrase-multilingual-mpnet-base-v2 (Multilingual, High Quality)",
    "all-mpnet-base-v2": "all-mpnet-base-v2 (English, High Quality)",
}


def _create_llm_provider(provider: str, model: str | None = None) -> LLMProvider:
    """Factory: LLM provider oluşturur."""
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIProvider(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    elif provider == "gemini":
        return GeminiProvider(model=model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    elif provider == "claude":
        return ClaudeProvider(model=model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    elif provider == "ollama":
        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
        )
    elif provider == "vllm":
        return VLLMProvider(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
            model=model or os.getenv("VLLM_MODEL", ""),
        )
    else:
        raise LLMConfigError(
            f"Bilinmeyen LLM provider: '{provider}'.\n"
            "Desteklenen: openai, gemini, claude, ollama, vllm"
        )


def _create_vector_store(store_type: str, collection_name: str = "rag_demo") -> VectorStore:
    """Factory: Vector store oluşturur."""
    store_type = store_type.lower().strip()
    if store_type == "chroma":
        return ChromaVectorStore(collection_name=collection_name)
    elif store_type == "faiss":
        return FAISSVectorStore(collection_name=collection_name)
    else:
        raise VectorStoreError(
            f"Bilinmeyen vector store: '{store_type}'.\n"
            "Desteklenen: chroma, faiss"
        )


# ---------------------------------------------------------------------------
# ClassicalRAGPipeline
# ---------------------------------------------------------------------------


class ClassicalRAGPipeline:
    """Esnek, provider-agnostic RAG pipeline.

    LLM: OpenAI, Gemini, Claude, Ollama, vLLM
    Vector Store: ChromaDB, FAISS
    Embedding: sentence-transformers (configurable)
    """

    SUPPORTED_EXTENSIONS = {"pdf", "txt", "csv", "xlsx", "xls", "docx"}

    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str | None = None,
        vector_store: str = "chroma",
        collection_name: str = "rag_demo",
    ) -> None:
        # LLM
        self._llm = _create_llm_provider(llm_provider, model_name)
        self._provider_name = llm_provider
        self._model_name = model_name or getattr(self._llm, "model", "unknown")

        # Embedding
        from chunking import _load_model, get_device
        embed_short = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        embed_display = _EMBEDDING_MODELS.get(embed_short, f"{embed_short} (Custom)")
        device = get_device()
        self._tokenizer, self._embedding_model = _load_model(embed_display, device)

        # Vector Store
        self._store = _create_vector_store(vector_store, collection_name)

        logger.info(
            "Pipeline: provider=%s, model=%s, vector_store=%s, embedding=%s",
            llm_provider, self._model_name, vector_store, embed_short,
        )

    def ingest(self, file_paths: List[str]) -> int:
        from chunking import (
            read_pdf_content, read_txt_content, read_tabular_data,
            sentence_chunk_with_overlap, create_chunk_record, normalize_whitespace,
        )
        from config import APP_CONFIG

        defaults = APP_CONFIG["default_values"]
        max_tokens, overlap = defaults["max_tokens"], defaults["overlap"]
        total = 0

        for fpath in file_paths:
            path = Path(fpath)
            if not path.exists():
                logger.warning("Dosya bulunamadı: %s", fpath)
                continue
            ext = path.suffix.lstrip(".").lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning("Desteklenmeyen tür: %s", ext)
                continue

            logger.info("İşleniyor: %s", path.name)
            try:
                if ext in ("csv", "xlsx", "xls"):
                    chunks = self._ingest_tabular(path, ext, max_tokens, overlap)
                else:
                    chunks = self._ingest_text(path, ext, max_tokens, overlap)
            except Exception as exc:
                logger.error("Hata (%s): %s", path.name, exc)
                continue

            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False).tolist()
            ids = [c["chunk_id"] for c in chunks]
            metadatas = [
                {"source_file": c.get("file", path.name), "source_type": ext,
                 "chunk_index": c.get("part", 0), "file_type": ext}
                for c in chunks
            ]

            self._store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            total += len(chunks)
            logger.info("  → %d chunk kaydedildi (%s)", len(chunks), path.name)

        logger.info("Toplam %d chunk kaydedildi.", total)
        return total

    def _ingest_text(self, path, ext, max_tokens, overlap):
        from chunking import read_pdf_content, read_txt_content, sentence_chunk_with_overlap, create_chunk_record, normalize_whitespace
        with open(path, "rb") as f:
            if ext == "pdf":
                raw = read_pdf_content(f)
            elif ext == "txt":
                raw = read_txt_content(f)
            elif ext == "docx":
                from chunking import read_docx_content
                raw = read_docx_content(f)
            else:
                raw = ""
        if not raw or not raw.strip():
            return []
        text = normalize_whitespace(raw)
        chunk_texts = sentence_chunk_with_overlap(text, self._tokenizer, max_tokens, overlap)
        return [create_chunk_record(file_name=path.name, source_type=ext, text=ct, part_index=i) for i, ct in enumerate(chunk_texts)]

    def _ingest_tabular(self, path, ext, max_tokens, overlap):
        from chunking import read_tabular_data, sentence_chunk_with_overlap, create_chunk_record, normalize_whitespace
        with open(path, "rb") as f:
            df = read_tabular_data(f, ext)
        if df is None or df.empty:
            return []
        records = []
        for row_idx, row in df.iterrows():
            row_text = normalize_whitespace(" | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val)))
            if not row_text.strip():
                continue
            chunk_texts = sentence_chunk_with_overlap(row_text, self._tokenizer, max_tokens, overlap)
            for pi, ct in enumerate(chunk_texts):
                records.append(create_chunk_record(file_name=path.name, source_type=ext, text=ct, part_index=pi, row_index=int(row_idx)))
        return records

    def get_relevant_chunks(self, question: str, top_k: int = 5) -> List[ChunkResult]:
        qe = self._embedding_model.encode([question], show_progress_bar=False).tolist()[0]
        return self._store.query(qe, top_k)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        chunks = self.get_relevant_chunks(question, top_k)
        if not chunks:
            return RAGResponse(answer="İlgili bilgi bulunamadı.", sources=[], model_used=self._model_name, provider=self._provider_name)

        context = "\n\n".join(
            f"[Kaynak {i}: {c.metadata.get('source_file', '?')}]\n{c.text}"
            for i, c in enumerate(chunks, 1)
        )
        answer = self._llm.generate(question, context)
        return RAGResponse(answer=answer, sources=chunks, model_used=self._model_name, provider=self._provider_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"
_DIM = "\033[2m"


def _print_banner():
    print(f"""
{_BOLD}{'=' * 60}
  📚 Classical RAG Demo Pipeline
  Çoklu LLM + Çoklu Vector Store
{'=' * 60}{_RESET}
""")


def _print_sources(sources: List[ChunkResult]):
    if not sources:
        return
    print(f"\n{_DIM}{'─' * 50}")
    print(f"  📎 Kaynaklar ({len(sources)} chunk):")
    for i, src in enumerate(sources, 1):
        sf = src.metadata.get("source_file", "?")
        preview = src.text[:120].replace("\n", " ")
        print(f"  {i}. [{sf}] (skor: {src.score:.3f})")
        print(f"     {preview}{'…' if len(src.text) > 120 else ''}")
    print(f"{'─' * 50}{_RESET}")


def _collect_files(data_dir: str) -> List[str]:
    dp = Path(data_dir)
    if not dp.exists():
        return []
    return [str(f) for f in sorted(dp.iterdir()) if f.is_file() and f.suffix.lstrip(".").lower() in ClassicalRAGPipeline.SUPPORTED_EXTENSIONS]


def main():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Classical RAG Demo Pipeline")
    parser.add_argument("--provider", choices=["openai", "gemini", "claude", "ollama", "vllm"],
                        default=os.getenv("LLM_PROVIDER", "openai"), help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model adı")
    parser.add_argument("--vector-store", choices=["chroma", "faiss"],
                        default=os.getenv("VECTOR_STORE", "chroma"), help="Vector store")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "examples" / "data"), help="Veri dizini")
    parser.add_argument("--collection", default="rag_demo", help="Koleksiyon adı")
    args = parser.parse_args()

    _print_banner()

    try:
        print(f"{_CYAN}🔧 Pipeline başlatılıyor (provider={args.provider}, store={args.vector_store})...{_RESET}")
        pipeline = ClassicalRAGPipeline(
            llm_provider=args.provider, model_name=args.model,
            vector_store=args.vector_store, collection_name=args.collection,
        )
        print(f"{_GREEN}✓ Pipeline hazır.{_RESET}\n")
    except (LLMConfigError, LLMConnectionError, VectorStoreError) as exc:
        print(f"\n{_RED}❌ Hata: {exc}{_RESET}")
        sys.exit(1)

    files = _collect_files(args.data_dir)
    if files:
        print(f"{_CYAN}📂 {len(files)} dosya bulundu:{_RESET}")
        for f in files:
            print(f"   • {Path(f).name}")
        try:
            n = pipeline.ingest(files)
            print(f"{_GREEN}✓ {n} chunk kaydedildi.{_RESET}\n")
        except VectorStoreError as exc:
            print(f"{_RED}❌ {exc}{_RESET}")
            sys.exit(1)
    else:
        print(f"{_YELLOW}⚠ Dosya bulunamadı: {args.data_dir}{_RESET}\n")

    print(f"{_BOLD}💬 Soru-Cevap (çıkmak için 'q'){_RESET}\n")

    try:
        while True:
            try:
                question = input(f"{_BOLD}Soru: {_RESET}").strip()
            except EOFError:
                break
            if not question:
                continue
            if question.lower() in ("q", "quit", "exit", "çık"):
                break
            try:
                resp = pipeline.query(question)
            except RAGPipelineError as exc:
                print(f"{_RED}❌ {exc}{_RESET}\n")
                continue
            print(f"\n{_GREEN}{_BOLD}Cevap ({resp.provider}/{resp.model_used}):{_RESET}")
            print(resp.answer)
            _print_sources(resp.sources)
            print()
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}👋 Çıkılıyor...{_RESET}")

    print(f"{_DIM}Demo tamamlandı.{_RESET}")


if __name__ == "__main__":
    main()

"""
Classical RAG Demo Pipeline
============================
Mevcut chunking.py modülü üzerine inşa edilen uçtan uca RAG pipeline.
ChromaDB vektör depolama, benzerlik araması ve LLM ile cevap üretme katmanlarını ekler.

Kullanım:
    python demo_pipeline.py --provider openai --data-dir ../examples/data
    python demo_pipeline.py --provider ollama --model llama3.2

Genişletilebilirlik Notları:
    - Bu demo'da OCR adımı uygulanmamıştır. Taranmış belgeler için PaddleOCR,
      Tesseract veya bulut Vision modelleri (Gemini, Claude Vision) entegre edilebilir.
    - Bu demo'da Reranking adımı uygulanmamıştır. bge-reranker-v2-m3 veya
      Cohere Rerank 3.5 gibi reranker'lar ekleyerek arama kalitesini artırabilirsiniz.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Custom exception hierarchy for user-friendly error messages
# ---------------------------------------------------------------------------


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors."""


class LLMConfigError(RAGPipelineError):
    """Raised when LLM configuration is invalid (missing API key, bad model name)."""


class LLMConnectionError(RAGPipelineError):
    """Raised when LLM service is unreachable."""


class VectorStoreError(RAGPipelineError):
    """Raised when ChromaDB connection or operation fails."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("demo_pipeline")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    """A single retrieved chunk with its similarity score and metadata."""

    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    sources: List[ChunkResult]
    model_used: str
    provider: str


# ---------------------------------------------------------------------------
# Task 2.2 – LLM Provider Protocol & Implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that every LLM provider must satisfy."""

    def generate(self, prompt: str, context: str) -> str:
        """Generate an answer given a prompt and retrieved context."""
        ...


class OpenAIProvider:
    """OpenAI API üzerinden cevap üretimi.

    Requires the ``OPENAI_API_KEY`` environment variable (or explicit *api_key*).
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key or resolved_key == "your-api-key-here":
            raise LLMConfigError(
                "OpenAI API anahtarı bulunamadı veya geçersiz.\n"
                "Lütfen .env dosyanızda OPENAI_API_KEY değerini ayarlayın.\n"
                "Örnek:  OPENAI_API_KEY=sk-..."
            )
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:
            raise LLMConfigError(
                "openai paketi yüklü değil. Lütfen çalıştırın: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=resolved_key)
        self.model = model
        self.provider_name = "openai"

    def generate(self, prompt: str, context: str) -> str:
        """Send prompt + context to OpenAI and return the assistant message."""
        system_msg = (
            "Sen yardımcı bir asistansın. Sana verilen bağlam bilgisini kullanarak "
            "soruları yanıtla. Bağlamda bilgi yoksa, bilmediğini belirt."
        )
        user_msg = f"Bağlam:\n{context}\n\nSoru: {prompt}"
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            error_str = str(exc).lower()
            if "api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise LLMConfigError(
                    "OpenAI API anahtarı geçersiz. Lütfen .env dosyanızı kontrol edin."
                ) from exc
            if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                raise LLMConfigError(
                    f"Model '{self.model}' bulunamadı. Desteklenen modeller: gpt-4o-mini, gpt-4o, gpt-3.5-turbo"
                ) from exc
            raise LLMConnectionError(
                f"OpenAI API isteği başarısız oldu: {exc}"
            ) from exc


class OllamaProvider:
    """Ollama yerel model üzerinden cevap üretimi.

    Ollama servisinin çalışıyor olması gerekir (``ollama serve``).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.provider_name = "ollama"
        # Verify connectivity eagerly so the user gets fast feedback.
        self._check_connection()

    def _check_connection(self) -> None:
        """Ping the Ollama server to verify it is reachable."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, OSError, ConnectionError) as exc:
            raise LLMConnectionError(
                f"Ollama servisine bağlanılamadı ({self.base_url}).\n"
                "Lütfen Ollama'nın çalıştığından emin olun:\n"
                "  ollama serve\n"
                "Veya Docker ile:\n"
                "  docker-compose up ollama"
            ) from exc

    def generate(self, prompt: str, context: str) -> str:
        """Send prompt + context to Ollama and return the response."""
        try:
            import ollama as ollama_lib  # lazy import
        except ImportError as exc:
            raise LLMConfigError(
                "ollama paketi yüklü değil. Lütfen çalıştırın: pip install ollama"
            ) from exc

        system_msg = (
            "Sen yardımcı bir asistansın. Sana verilen bağlam bilgisini kullanarak "
            "soruları yanıtla. Bağlamda bilgi yoksa, bilmediğini belirt."
        )
        user_msg = f"Bağlam:\n{context}\n\nSoru: {prompt}"
        try:
            client = ollama_lib.Client(host=self.base_url)
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            return response["message"]["content"]
        except Exception as exc:
            error_str = str(exc).lower()
            if "not found" in error_str or "pull" in error_str:
                raise LLMConfigError(
                    f"Model '{self.model}' Ollama'da bulunamadı.\n"
                    f"Lütfen modeli indirin: ollama pull {self.model}"
                ) from exc
            raise LLMConnectionError(
                f"Ollama isteği başarısız oldu: {exc}\n"
                "Ollama servisinin çalıştığından emin olun: ollama serve"
            ) from exc


# ---------------------------------------------------------------------------
# Task 2.3 – ClassicalRAGPipeline
# ---------------------------------------------------------------------------

# Embedding model used for multilingual (Turkish) support
_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)"


def _create_llm_provider(provider: str, model: str | None = None) -> LLMProvider:
    """Factory that instantiates the requested LLM provider with error handling."""
    if provider == "openai":
        return OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    elif provider == "ollama":
        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
        )
    else:
        raise LLMConfigError(
            f"Bilinmeyen LLM provider: '{provider}'. Desteklenen: openai, ollama"
        )


class ClassicalRAGPipeline:
    """Mevcut chunking modülü üzerine inşa edilen uçtan uca RAG pipeline.

    Genişletilebilirlik Notları
    ---------------------------
    - Bu demo'da OCR adımı uygulanmamıştır. Taranmış belgeler için PaddleOCR,
      Tesseract veya bulut Vision modelleri entegre edilebilir.
    - Bu demo'da Reranking adımı uygulanmamıştır. bge-reranker-v2-m3 veya
      Cohere Rerank 3.5 gibi reranker'lar ekleyerek arama kalitesini artırabilirsiniz.
    """

    SUPPORTED_EXTENSIONS = {"pdf", "txt", "csv", "xlsx", "xls", "docx"}

    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str | None = None,
        chroma_persist_dir: str | None = None,
        collection_name: str = "rag_demo",
    ) -> None:
        """
        Args:
            llm_provider: ``"openai"`` veya ``"ollama"``
            model_name: LLM model adı (varsayılan: provider'a göre)
            chroma_persist_dir: ChromaDB kalıcı depolama dizini
            collection_name: ChromaDB koleksiyon adı
        """
        # --- LLM provider ---
        self._llm: LLMProvider = _create_llm_provider(llm_provider, model_name)
        self._provider_name = llm_provider
        self._model_name = model_name or (
            getattr(self._llm, "model", "unknown")
        )

        # --- Embedding model (from existing chunking module) ---
        from chunking import _load_model, get_device

        device = get_device()
        self._tokenizer, self._embedding_model = _load_model(
            _EMBEDDING_MODEL_NAME, device
        )

        # --- ChromaDB ---
        persist_dir = chroma_persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb paketi yüklü değil. Lütfen çalıştırın: pip install chromadb"
            ) from exc

        try:
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB bağlantısı kurulamadı ({persist_dir}).\n"
                "Lütfen ChromaDB'nin erişilebilir olduğundan emin olun.\n"
                "Docker ile: docker-compose up chromadb"
            ) from exc

        logger.info(
            "Pipeline başlatıldı – provider=%s, model=%s, collection=%s",
            llm_provider,
            self._model_name,
            collection_name,
        )

    # ---- Ingest ----

    def ingest(self, file_paths: List[str]) -> int:
        """Belgeleri işleyip ChromaDB'ye kaydeder.

        Mevcut chunking modülünün alt seviye fonksiyonlarını kullanarak
        Streamlit bağımlılığı olmadan çalışır.

        Returns:
            Kaydedilen toplam chunk sayısı.
        """
        from chunking import (
            read_pdf_content,
            read_txt_content,
            read_tabular_data,
            sentence_chunk_with_overlap,
            create_chunk_record,
            normalize_whitespace,
            get_token_count,
        )
        from config import APP_CONFIG

        defaults = APP_CONFIG["default_values"]
        max_tokens = defaults["max_tokens"]
        overlap = defaults["overlap"]

        total_chunks = 0

        for fpath in file_paths:
            path = Path(fpath)
            if not path.exists():
                logger.warning("Dosya bulunamadı, atlanıyor: %s", fpath)
                continue

            ext = path.suffix.lstrip(".").lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning("Desteklenmeyen dosya türü, atlanıyor: %s", ext)
                continue

            file_name = path.name
            logger.info("İşleniyor: %s", file_name)

            try:
                if ext in ("csv", "xlsx", "xls"):
                    chunks = self._ingest_tabular(path, ext, max_tokens, overlap)
                else:
                    chunks = self._ingest_text(path, ext, max_tokens, overlap)
            except Exception as exc:
                logger.error("Dosya işlenirken hata (%s): %s", file_name, exc)
                continue

            if not chunks:
                logger.warning("Chunk üretilemedi: %s", file_name)
                continue

            # Generate embeddings and store in ChromaDB
            texts = [c["text"] for c in chunks]
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False).tolist()

            ids = [c["chunk_id"] for c in chunks]
            metadatas = [
                {
                    "source_file": c.get("file", file_name),
                    "source_type": c.get("source", ext),
                    "chunk_index": c.get("part", 0),
                    "file_type": ext,
                }
                for c in chunks
            ]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            logger.info("  → %d chunk kaydedildi (%s)", len(chunks), file_name)

        logger.info("Toplam %d chunk ChromaDB'ye kaydedildi.", total_chunks)
        return total_chunks

    # ---- Internal helpers for ingest ----

    def _ingest_text(
        self, path: Path, ext: str, max_tokens: int, overlap: int
    ) -> List[Dict[str, Any]]:
        """Process a text-based document (PDF, TXT, DOCX) into chunks."""
        from chunking import (
            read_pdf_content,
            read_txt_content,
            sentence_chunk_with_overlap,
            create_chunk_record,
            normalize_whitespace,
        )

        with open(path, "rb") as f:
            if ext == "pdf":
                raw_text = read_pdf_content(f)
            elif ext == "txt":
                raw_text = read_txt_content(f)
            elif ext == "docx":
                from chunking import read_docx_content
                raw_text = read_docx_content(f)
            else:
                raw_text = ""

        if not raw_text or not raw_text.strip():
            return []

        text = normalize_whitespace(raw_text)
        chunk_texts = sentence_chunk_with_overlap(
            text, self._tokenizer, max_tokens, overlap
        )

        records = []
        for idx, chunk_text in enumerate(chunk_texts):
            rec = create_chunk_record(
                file_name=path.name,
                source_type=ext,
                text=chunk_text,
                part_index=idx,
            )
            records.append(rec)
        return records

    def _ingest_tabular(
        self, path: Path, ext: str, max_tokens: int, overlap: int
    ) -> List[Dict[str, Any]]:
        """Process a tabular document (CSV, Excel) into chunks."""
        from chunking import (
            read_tabular_data,
            sentence_chunk_with_overlap,
            create_chunk_record,
            normalize_whitespace,
        )

        with open(path, "rb") as f:
            df = read_tabular_data(f, ext)

        if df is None or df.empty:
            return []

        records = []
        for row_idx, row in df.iterrows():
            row_text = " | ".join(
                f"{col}: {val}" for col, val in row.items() if pd.notna(val)
            )
            row_text = normalize_whitespace(row_text)
            if not row_text.strip():
                continue

            chunk_texts = sentence_chunk_with_overlap(
                row_text, self._tokenizer, max_tokens, overlap
            )
            for part_idx, chunk_text in enumerate(chunk_texts):
                rec = create_chunk_record(
                    file_name=path.name,
                    source_type=ext,
                    text=chunk_text,
                    part_index=part_idx,
                    row_index=int(row_idx),
                )
                records.append(rec)
        return records

    # ---- Retrieval ----

    def get_relevant_chunks(
        self, question: str, top_k: int = 5
    ) -> List[ChunkResult]:
        """Sadece benzerlik araması yapar (LLM çağrısı olmadan).

        Returns:
            Benzerlik skoruna göre sıralı chunk listesi.
        """
        query_embedding = self._embedding_model.encode(
            [question], show_progress_bar=False
        ).tolist()

        try:
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self._collection.count() or top_k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB sorgusu başarısız oldu: {exc}\n"
                "ChromaDB'nin çalıştığından emin olun: docker-compose up chromadb"
            ) from exc

        chunks: List[ChunkResult] = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance → similarity score
                score = 1.0 - dist
                chunks.append(ChunkResult(text=doc, score=score, metadata=meta))

        return chunks

    # ---- Query (retrieval + generation) ----

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Benzerlik araması yapıp LLM ile cevap üretir.

        Returns:
            RAGResponse with answer, sources, model info.
        """
        chunks = self.get_relevant_chunks(question, top_k)

        if not chunks:
            return RAGResponse(
                answer="İlgili bilgi bulunamadı. Lütfen önce belge yükleyin (ingest).",
                sources=[],
                model_used=self._model_name,
                provider=self._provider_name,
            )

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source_file", "bilinmeyen")
            context_parts.append(f"[Kaynak {i}: {source}]\n{chunk.text}")
        context = "\n\n".join(context_parts)

        answer = self._llm.generate(question, context)

        return RAGResponse(
            answer=answer,
            sources=chunks,
            model_used=self._model_name,
            provider=self._provider_name,
        )


# ---------------------------------------------------------------------------
# Task 2.4 – main() CLI demo
# ---------------------------------------------------------------------------

# ANSI color helpers for terminal output
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"
_DIM = "\033[2m"


def _print_banner() -> None:
    print(f"""
{_BOLD}{'=' * 60}
  📚 Classical RAG Demo Pipeline
  Mevcut chunking modülü + ChromaDB + LLM
{'=' * 60}{_RESET}
""")


def _print_sources(sources: List[ChunkResult]) -> None:
    """Print retrieved sources with colored formatting."""
    if not sources:
        return
    print(f"\n{_DIM}{'─' * 50}")
    print(f"  📎 Kaynaklar ({len(sources)} chunk):")
    for i, src in enumerate(sources, 1):
        source_file = src.metadata.get("source_file", "?")
        print(f"  {i}. [{source_file}] (skor: {src.score:.3f})")
        # Show a short preview of the chunk text
        preview = src.text[:120].replace("\n", " ")
        if len(src.text) > 120:
            preview += "…"
        print(f"     {preview}")
    print(f"{'─' * 50}{_RESET}")


def _collect_files(data_dir: str) -> List[str]:
    """Collect all supported files from a directory."""
    supported = ClassicalRAGPipeline.SUPPORTED_EXTENSIONS
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Veri dizini bulunamadı: %s", data_dir)
        return []

    files = []
    for f in sorted(data_path.iterdir()):
        if f.is_file() and f.suffix.lstrip(".").lower() in supported:
            files.append(str(f))
    return files


def main() -> None:
    """CLI entry point: load data, ingest, interactive Q&A loop."""
    # Load environment variables from .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    # Also try local .env
    load_dotenv()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Classical RAG Demo Pipeline – ChromaDB + LLM",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider (varsayılan: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model adı (varsayılan: provider'a göre)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "examples" / "data"),
        help="Örnek veri dizini (varsayılan: examples/data)",
    )
    parser.add_argument(
        "--collection",
        default="rag_demo",
        help="ChromaDB koleksiyon adı (varsayılan: rag_demo)",
    )
    args = parser.parse_args()

    _print_banner()

    # --- Initialize pipeline with error handling (Task 2.5) ---
    try:
        print(f"{_CYAN}🔧 Pipeline başlatılıyor (provider={args.provider})...{_RESET}")
        pipeline = ClassicalRAGPipeline(
            llm_provider=args.provider,
            model_name=args.model,
            collection_name=args.collection,
        )
        print(f"{_GREEN}✓ Pipeline hazır.{_RESET}\n")
    except LLMConfigError as exc:
        print(f"\n{_RED}❌ LLM Yapılandırma Hatası:{_RESET}")
        print(f"   {exc}")
        sys.exit(1)
    except LLMConnectionError as exc:
        print(f"\n{_RED}❌ LLM Bağlantı Hatası:{_RESET}")
        print(f"   {exc}")
        sys.exit(1)
    except VectorStoreError as exc:
        print(f"\n{_RED}❌ ChromaDB Hatası:{_RESET}")
        print(f"   {exc}")
        sys.exit(1)

    # --- Ingest example data ---
    files = _collect_files(args.data_dir)
    if files:
        print(f"{_CYAN}📂 {len(files)} dosya bulundu ({args.data_dir}):{_RESET}")
        for f in files:
            print(f"   • {Path(f).name}")
        print()

        try:
            n = pipeline.ingest(files)
            print(f"{_GREEN}✓ {n} chunk başarıyla ChromaDB'ye kaydedildi.{_RESET}\n")
        except VectorStoreError as exc:
            print(f"{_RED}❌ Veri yükleme hatası: {exc}{_RESET}")
            sys.exit(1)
    else:
        print(
            f"{_YELLOW}⚠ Veri dizininde desteklenen dosya bulunamadı: {args.data_dir}\n"
            f"  Dosya ekledikten sonra tekrar çalıştırın.{_RESET}\n"
        )

    # --- Interactive Q&A loop ---
    print(f"{_BOLD}💬 Soru-Cevap Modu (çıkmak için 'q' veya Ctrl+C){_RESET}")
    print(f"{_DIM}   Bir soru yazın ve Enter'a basın.{_RESET}\n")

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
                response = pipeline.query(question)
            except LLMConnectionError as exc:
                print(f"{_RED}❌ LLM hatası: {exc}{_RESET}\n")
                continue
            except VectorStoreError as exc:
                print(f"{_RED}❌ Arama hatası: {exc}{_RESET}\n")
                continue
            except RAGPipelineError as exc:
                print(f"{_RED}❌ Pipeline hatası: {exc}{_RESET}\n")
                continue

            # Print answer
            print(f"\n{_GREEN}{_BOLD}Cevap ({response.provider}/{response.model_used}):{_RESET}")
            print(response.answer)

            # Print sources
            _print_sources(response.sources)
            print()

    except KeyboardInterrupt:
        print(f"\n\n{_YELLOW}👋 Çıkılıyor...{_RESET}")

    print(f"{_DIM}Demo tamamlandı.{_RESET}")


if __name__ == "__main__":
    main()

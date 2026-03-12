"""
Agentic RAG Tool Definitions
=============================
OpenAI function calling için araç tanımları.
VectorSearchTool: ChromaDB veya FAISS üzerinde benzerlik araması.
WebSearchTool: Web arama simülasyonu (demo amaçlı).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("agentic_tools")

_AGENTIC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTIC_DIR.parent
_CLASSICAL_RAG_DIR = _PROJECT_ROOT / "Classical-RAG"

if str(_CLASSICAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSICAL_RAG_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Embedding model loader
# ---------------------------------------------------------------------------

_EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (English, Fast)",
}


def _load_embedding_model():
    from chunking import _load_model, get_device
    embed_short = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    embed_display = _EMBEDDING_MODELS.get(embed_short, f"{embed_short} (Custom)")
    device = get_device()
    _, model = _load_model(embed_display, device)
    return model


# ---------------------------------------------------------------------------
# VectorSearchTool
# ---------------------------------------------------------------------------


class VectorSearchTool:
    """ChromaDB veya FAISS üzerinde vektör benzerlik araması."""

    TOOL_DEFINITION: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Yerel vektör veritabanında anlamsal benzerlik araması yapar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Aranacak soru veya konu"},
                    "top_k": {"type": "integer", "description": "Maks sonuç sayısı", "default": 3},
                },
                "required": ["query"],
            },
        },
    }

    def __init__(
        self,
        collection_name: str = "rag_demo",
        chroma_persist_dir: Optional[str] = None,
    ) -> None:
        self._embed_model = _load_embedding_model()
        vector_store = os.getenv("VECTOR_STORE", "chroma").lower()

        if vector_store == "faiss":
            self._store_type = "faiss"
            self._init_faiss(collection_name)
        else:
            self._store_type = "chroma"
            self._init_chroma(collection_name, chroma_persist_dir)

        logger.info("VectorSearchTool hazır – store=%s, collection=%s", self._store_type, collection_name)

    def _init_chroma(self, collection_name, persist_dir):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        import chromadb
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"},
        )

    def _init_faiss(self, collection_name):
        import faiss
        self._faiss = faiss
        self._faiss_dir = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
        self._collection_name = collection_name
        self._faiss_index = None
        self._faiss_docs: List[str] = []
        self._faiss_metas: List[Dict] = []
        self._faiss_ids: List[str] = []
        idx_path = os.path.join(self._faiss_dir, f"{collection_name}.index")
        meta_path = os.path.join(self._faiss_dir, f"{collection_name}.meta.npz")
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self._faiss_index = faiss.read_index(idx_path)
            data = np.load(meta_path, allow_pickle=True)
            self._faiss_docs = data["documents"].tolist()
            self._faiss_metas = [json.loads(m) for m in data["metadatas"].tolist()]
            self._faiss_ids = data["ids"].tolist()

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        qe = self._embed_model.encode([query], show_progress_bar=False).tolist()

        if self._store_type == "faiss":
            return self._search_faiss(qe[0], top_k)
        return self._search_chroma(qe, top_k)

    def _search_chroma(self, qe, top_k):
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=qe, n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                out.append(SearchResult(text=doc, score=1.0 - dist, source=meta.get("source_file", "?"), metadata=meta))
        return out

    def _search_faiss(self, qe, top_k):
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        qv = np.array([qe], dtype=np.float32)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        k = min(top_k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(qv, k)
        out = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._faiss_docs):
                out.append(SearchResult(
                    text=self._faiss_docs[idx], score=float(score),
                    source=self._faiss_metas[idx].get("source_file", "?"),
                    metadata=self._faiss_metas[idx],
                ))
        return out

    def ingest_directory(self, data_dir: str) -> int:
        """Dizindeki dosyaları vektör store'a yükler."""
        from chunking import (
            read_pdf_content, read_txt_content, read_tabular_data,
            sentence_chunk_with_overlap, create_chunk_record, normalize_whitespace,
        )
        from config import APP_CONFIG

        defaults = APP_CONFIG["default_values"]
        max_tokens, overlap = defaults["max_tokens"], defaults["overlap"]
        from chunking import _load_model, get_device
        embed_short = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        embed_display = _EMBEDDING_MODELS.get(embed_short, f"{embed_short} (Custom)")
        device = get_device()
        tokenizer, _ = _load_model(embed_display, device)

        supported = {"pdf", "txt", "csv", "xlsx", "xls", "docx"}
        data_path = Path(data_dir)
        total = 0

        for fpath in sorted(data_path.iterdir()):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lstrip(".").lower()
            if ext not in supported:
                continue

            try:
                if ext in ("csv", "xlsx", "xls"):
                    import pandas as pd
                    with open(fpath, "rb") as f:
                        df = read_tabular_data(f, ext)
                    if df is None or df.empty:
                        continue
                    chunks = []
                    for ri, row in df.iterrows():
                        rt = normalize_whitespace(" | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v)))
                        if rt.strip():
                            for pi, ct in enumerate(sentence_chunk_with_overlap(rt, tokenizer, max_tokens, overlap)):
                                chunks.append(create_chunk_record(file_name=fpath.name, source_type=ext, text=ct, part_index=pi))
                else:
                    with open(fpath, "rb") as f:
                        if ext == "pdf":
                            raw = read_pdf_content(f)
                        elif ext == "txt":
                            raw = read_txt_content(f)
                        elif ext == "docx":
                            from chunking import read_docx_content
                            raw = read_docx_content(f)
                        else:
                            continue
                    if not raw or not raw.strip():
                        continue
                    text = normalize_whitespace(raw)
                    chunks = [create_chunk_record(file_name=fpath.name, source_type=ext, text=ct, part_index=i)
                              for i, ct in enumerate(sentence_chunk_with_overlap(text, tokenizer, max_tokens, overlap))]

                if not chunks:
                    continue

                texts = [c["text"] for c in chunks]
                embeddings = self._embed_model.encode(texts, show_progress_bar=False).tolist()
                ids = [c["chunk_id"] for c in chunks]
                metas = [{"source_file": c.get("file", fpath.name), "source_type": ext} for c in chunks]

                if self._store_type == "chroma":
                    self._collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
                else:
                    self._faiss_ingest(ids, embeddings, texts, metas)

                total += len(chunks)
            except Exception as exc:
                logger.error("Dosya hatası (%s): %s", fpath.name, exc)

        return total

    def _faiss_ingest(self, ids, embeddings, documents, metadatas):
        vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        if self._faiss_index is None:
            self._faiss_index = self._faiss.IndexFlatIP(vectors.shape[1])

        self._faiss_index.add(vectors)
        self._faiss_ids.extend(ids)
        self._faiss_docs.extend(documents)
        self._faiss_metas.extend(metadatas)

        os.makedirs(self._faiss_dir, exist_ok=True)
        self._faiss.write_index(self._faiss_index, os.path.join(self._faiss_dir, f"{self._collection_name}.index"))
        np.savez(
            os.path.join(self._faiss_dir, f"{self._collection_name}.meta.npz"),
            documents=np.array(self._faiss_docs, dtype=object),
            metadatas=np.array([json.dumps(m) for m in self._faiss_metas], dtype=object),
            ids=np.array(self._faiss_ids, dtype=object),
        )


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Web arama simülasyonu (demo amaçlı).

    Gerçek implementasyon için: SerpAPI, Tavily veya DuckDuckGo API kullanılabilir.
    """

    TOOL_DEFINITION: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "İnternette güncel bilgi arar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Aranacak konu"},
                },
                "required": ["query"],
            },
        },
    }

    _DEMO_RESPONSES: Dict[str, str] = {
        "rag": "RAG (Retrieval-Augmented Generation), LLM'lerin harici bilgi kaynaklarıyla zenginleştirilmesini sağlayan bir mimari yaklaşımdır.",
        "chromadb": "ChromaDB, Python-native açık kaynaklı bir vektör veritabanıdır.",
        "faiss": "FAISS (Facebook AI Similarity Search), Meta tarafından geliştirilen yüksek performanslı vektör arama kütüphanesidir.",
        "embedding": "Embedding modelleri, metni yüksek boyutlu vektörlere dönüştürür. Popüler: sentence-transformers, OpenAI text-embedding-3.",
        "llm": "Büyük Dil Modelleri: GPT-4, Claude, Gemini ve Llama popüler örneklerdir.",
        "vllm": "vLLM, yüksek throughput LLM inference engine'dir. PagedAttention ile verimli bellek yönetimi sağlar.",
    }

    def search(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, response in self._DEMO_RESPONSES.items():
            if keyword in query_lower:
                return f"[Web Arama - Demo]\n{response}"
        return (
            f"[Web Arama - Demo]\n'{query}' için arama yapıldı. "
            "Gerçek implementasyon için SerpAPI veya Tavily entegre edilebilir."
        )

"""
Agentic RAG Tool Definitions
=============================
OpenAI function calling icin arac tanimlari.
VectorSearchTool: ChromaDB uzerinde benzerlik aramasi.
WebSearchTool: Basit web arama simulasyonu (demo amacli).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_tools")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_AGENTIC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTIC_DIR.parent
_CLASSICAL_RAG_DIR = _PROJECT_ROOT / "Classical-RAG"

if str(_CLASSICAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSICAL_RAG_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Tek bir arama sonucu."""

    text: str
    score: float
    source: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# VectorSearchTool
# ---------------------------------------------------------------------------


class VectorSearchTool:
    """ChromaDB uzerinde vektor benzerlik aramasi yapan arac."""

    TOOL_DEFINITION: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Yerel vektor veritabaninda (ChromaDB) anlamsal benzerlik aramasi yapar. "
                "RAG bilgi tabanindaki belgelerde bilgi aramak icin kullanilir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Aranacak soru veya konu",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Dondurulecek maksimum sonuc sayisi (varsayilan: 3)",
                        "default": 3,
                    },
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
        persist_dir = chroma_persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB baglantisi kurulamadi ({persist_dir}): {exc}"
            ) from exc

        try:
            from chunking import _load_model, get_device
            device = get_device()
            _MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)"
            _, self._embed_model = _load_model(_MODEL_NAME, device)
        except Exception as exc:
            raise RuntimeError(f"Embedding modeli yuklenemedi: {exc}") from exc

        logger.info("VectorSearchTool hazir – collection=%s", collection_name)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Vektor benzerlik aramasi yapar."""
        query_embedding = self._embed_model.encode(
            [query], show_progress_bar=False
        ).tolist()

        count = self._collection.count()
        if count == 0:
            logger.warning("ChromaDB koleksiyonu bos. Once veri yukleyin.")
            return []

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1.0 - dist
                search_results.append(
                    SearchResult(
                        text=doc,
                        score=score,
                        source=meta.get("source_file", "bilinmeyen"),
                        metadata=meta,
                    )
                )

        return search_results


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Web arama simulasyonu (demo amacli).

    Gercek implementasyon icin: SerpAPI, Tavily veya DuckDuckGo API kullanilabilir.
    """

    TOOL_DEFINITION: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Internette guncel bilgi arar. Yerel veritabaninda bulunamayan "
                "guncel konular veya genel bilgi icin kullanilir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Aranacak konu veya soru",
                    },
                },
                "required": ["query"],
            },
        },
    }

    _DEMO_RESPONSES: Dict[str, str] = {
        "rag": (
            "RAG (Retrieval-Augmented Generation), LLM'lerin harici bilgi kaynaklariyla "
            "zenginlestirilmesini saglayan bir mimari yaklasimdir. 2020 yilinda Meta AI "
            "tarafindan tanitilmistir."
        ),
        "chromadb": (
            "ChromaDB, Python-native acik kaynakli bir vektor veritabanidir. "
            "Sifir yapilandirma ile calisir ve yerel gelistirme icin idealdir."
        ),
        "embedding": (
            "Embedding modelleri, metni yuksek boyutlu vektorlere donusturur. "
            "Populer modeller: sentence-transformers, OpenAI text-embedding-3, Cohere embed-v4."
        ),
        "llm": (
            "Buyuk Dil Modelleri (LLM), transformer mimarisi uzerine kurulu dil modelleridir. "
            "GPT-4, Claude, Gemini ve Llama populer orneklerdir."
        ),
    }

    def search(self, query: str) -> str:
        """Web aramasi simule eder."""
        query_lower = query.lower()
        for keyword, response in self._DEMO_RESPONSES.items():
            if keyword in query_lower:
                logger.info("WebSearchTool: '%s' icin demo yanit donduruldu", query)
                return f"[Web Arama Sonucu - Demo]\n{response}"

        return (
            f"[Web Arama Sonucu - Demo]\n"
            f"'{query}' icin web aramasi yapildi. "
            "Bu demo'da gercek web aramasi yapilmamaktadir. "
            "Gercek implementasyon icin SerpAPI veya Tavily entegre edilebilir."
        )

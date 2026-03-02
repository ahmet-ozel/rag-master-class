"""
Agentic RAG Demo
=================
OpenAI function calling ile otonom arac secimi yapan RAG demo.

Kullanim:
    python agent_demo.py --data-dir ../examples/data
    python agent_demo.py --model gpt-4o-mini --max-iterations 5

Mimari:
    - LLM hangi araci kullanacagina otonom karar verir
    - VectorSearchTool: ChromaDB'de benzerlik aramasi
    - WebSearchTool: Web arama simulasyonu
    - Maksimum 5 iterasyon ile sonsuz dongu onlenir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


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

from tools import VectorSearchTool, WebSearchTool, SearchResult

logger = logging.getLogger("agent_demo")

# ANSI renk kodlari
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"
_DIM = "\033[2m"


# ---------------------------------------------------------------------------
# AgenticRAGDemo
# ---------------------------------------------------------------------------


class AgenticRAGDemo:
    """OpenAI function calling ile otonom arac secimi yapan Agentic RAG demo.

    LLM, soruyu analiz ederek hangi araci kullanacagina kendi karar verir:
    - vector_search: Yerel ChromaDB'de bilgi arar
    - web_search: Web'de guncel bilgi arar (demo simulasyonu)

    Maksimum iterasyon siniri ile sonsuz dongu onlenir.
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        collection_name: str = "rag_demo",
        chroma_persist_dir: Optional[str] = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY bulunamadi.\n"
                "Lutfen .env dosyanizda OPENAI_API_KEY degerini ayarlayin."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "openai paketi yuklu degil. Calistirin: pip install openai"
            ) from exc

        self._model = model
        self._vector_tool = VectorSearchTool(
            collection_name=collection_name,
            chroma_persist_dir=chroma_persist_dir,
        )
        self._web_tool = WebSearchTool()

        # OpenAI'a gonderilecek arac tanimlari
        self._tools = [
            VectorSearchTool.TOOL_DEFINITION,
            WebSearchTool.TOOL_DEFINITION,
        ]

        logger.info("AgenticRAGDemo hazir – model=%s", model)

    def ingest(self, data_dir: str) -> int:
        """Veri dizinindeki dosyalari ChromaDB'ye yukler."""
        return self._vector_tool.ingest_directory(data_dir)

    def run(self, question: str) -> str:
        """Soruyu ajan dongusuyle yanitlar.

        LLM hangi araci kullanacagina otonom karar verir.
        Maksimum MAX_ITERATIONS iterasyondan sonra dongu sonlanir.

        Returns:
            Nihai cevap metni
        """
        print(f"\n{_CYAN}{_BOLD}Soru:{_RESET} {question}")
        print(f"{_DIM}{'─' * 60}{_RESET}")

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Sen yardimci bir RAG asistanisin. Sorulari yanıtlamak icin "
                    "oncelikle vector_search aracini kullan. Yerel veritabaninda "
                    "yeterli bilgi bulamazsan web_search aracini kullan. "
                    "Her zaman Turkce yanit ver."
                ),
            },
            {"role": "user", "content": question},
        ]

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n{_YELLOW}[Iterasyon {iteration}/{self.MAX_ITERATIONS}]{_RESET}")

            # LLM'e gonder
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._tools,
                tool_choice="auto",
                temperature=0.3,
            )

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Arac cagrisi yok – nihai cevap
            if finish_reason == "stop" or not msg.tool_calls:
                answer = msg.content or "Cevap uretildi."
                print(f"\n{_GREEN}{_BOLD}Nihai Cevap:{_RESET}")
                print(answer)
                return answer

            # Arac cagrilarini isle
            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  {_MAGENTA}Arac:{_RESET} {tool_name}")
                print(f"  {_DIM}Parametreler: {tool_args}{_RESET}")

                tool_result = self._execute_tool(tool_name, tool_args)

                print(f"  {_GREEN}Sonuc:{_RESET} {str(tool_result)[:200]}{'...' if len(str(tool_result)) > 200 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                })

        # Maksimum iterasyona ulasildi
        print(f"\n{_YELLOW}Maksimum iterasyon ({self.MAX_ITERATIONS}) sinirına ulasildi.{_RESET}")
        # Son bir kez LLM'den cevap al
        final_response = self._client.chat.completions.create(
            model=self._model,
            messages=messages + [
                {"role": "user", "content": "Toplanan bilgilere dayanarak soruyu yanıtla."}
            ],
            temperature=0.3,
        )
        answer = final_response.choices[0].message.content or "Yeterli bilgi toplanamadi."
        print(f"\n{_GREEN}{_BOLD}Nihai Cevap:{_RESET}")
        print(answer)
        return answer

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Arac cagrisini gerceklestirir ve sonucu string olarak dondurur."""
        if tool_name == "vector_search":
            query = args.get("query", "")
            top_k = args.get("top_k", 3)
            results = self._vector_tool.search(query, top_k)

            if not results:
                return "Vektor veritabaninda ilgili bilgi bulunamadi."

            parts = []
            for i, r in enumerate(results, 1):
                parts.append(f"[Kaynak {i}: {r.source} (skor: {r.score:.3f})]\n{r.text}")
            return "\n\n".join(parts)

        elif tool_name == "web_search":
            query = args.get("query", "")
            return self._web_tool.search(query)

        else:
            return f"Bilinmeyen arac: {tool_name}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    print(f"""
{_BOLD}{'=' * 60}
  Agentic RAG Demo
  OpenAI Function Calling + ChromaDB + Web Search
{'=' * 60}{_RESET}
""")


def main() -> None:
    """CLI giris noktasi."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Agentic RAG Demo")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model adi")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "examples" / "data"),
        help="Ornek veri dizini",
    )
    parser.add_argument("--collection", default="rag_demo", help="ChromaDB koleksiyon adi")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maksimum ajan iterasyon sayisi",
    )
    args = parser.parse_args()

    _print_banner()

    # Demo sinifini baslat
    try:
        print(f"{_CYAN}Agentic RAG Demo baslatiliyor (model={args.model})...{_RESET}")
        demo = AgenticRAGDemo(
            model=args.model,
            collection_name=args.collection,
        )
        # Maksimum iterasyon guncelle
        AgenticRAGDemo.MAX_ITERATIONS = args.max_iterations
        print(f"{_GREEN}Demo hazir.{_RESET}\n")
    except (ValueError, ImportError, RuntimeError) as exc:
        print(f"{_RED}Baslama hatasi: {exc}{_RESET}")
        sys.exit(1)

    # Veri yukle
    data_path = Path(args.data_dir)
    if data_path.exists():
        print(f"{_CYAN}Veri yukleniyor: {args.data_dir}{_RESET}")
        try:
            n = demo.ingest(args.data_dir)
            print(f"{_GREEN}{n} chunk ChromaDB'ye yuklendi.{_RESET}\n")
        except Exception as exc:
            print(f"{_YELLOW}Veri yukleme atlandi: {exc}{_RESET}\n")
    else:
        print(f"{_YELLOW}Veri dizini bulunamadi: {args.data_dir}{_RESET}\n")

    # Interaktif soru-cevap dongusu
    print(f"{_BOLD}Soru-Cevap Modu (cikmak icin 'q'){_RESET}")
    print(f"{_DIM}Ajan, sorunuzu yanıtlamak icin uygun araci otomatik sececek.{_RESET}\n")

    try:
        while True:
            try:
                question = input(f"{_BOLD}Soru: {_RESET}").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in ("q", "quit", "exit"):
                break

            try:
                demo.run(question)
            except Exception as exc:
                print(f"{_RED}Hata: {exc}{_RESET}")
            print()

    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Cikiliyor...{_RESET}")

    print(f"{_DIM}Demo tamamlandi.{_RESET}")


if __name__ == "__main__":
    main()

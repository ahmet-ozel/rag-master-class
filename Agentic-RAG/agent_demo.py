"""
Agentic RAG Demo
=================
Çoklu LLM provider desteği ile otonom araç seçimi yapan RAG demo.

Kullanım:
    python agent_demo.py --provider openai --data-dir ../examples/data
    python agent_demo.py --provider gemini --model gemini-2.0-flash
    python agent_demo.py --provider claude --model claude-sonnet-4-20250514
    python agent_demo.py --provider vllm --model meta-llama/Llama-3.2-3B-Instruct

Mimari:
    - LLM hangi aracı kullanacağına otonom karar verir
    - VectorSearchTool: ChromaDB/FAISS'te benzerlik araması
    - WebSearchTool: Web arama simülasyonu
    - Maksimum 5 iterasyon ile sonsuz döngü önlenir
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

_AGENTIC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTIC_DIR.parent
_CLASSICAL_RAG_DIR = _PROJECT_ROOT / "Classical-RAG"

if str(_CLASSICAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSICAL_RAG_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import VectorSearchTool, WebSearchTool, SearchResult

logger = logging.getLogger("agent_demo")

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"
_DIM = "\033[2m"


# ---------------------------------------------------------------------------
# LLM Client abstraction for function calling
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = (
    "Sen yardımcı bir RAG asistanısın. Soruları yanıtlamak için "
    "öncelikle vector_search aracını kullan. Yerel veritabanında "
    "yeterli bilgi bulamazsan web_search aracını kullan. "
    "Her zaman Türkçe yanıt ver."
)


class _OpenAIFunctionCaller:
    """OpenAI / vLLM function calling client."""

    def __init__(self, provider: str, model: str, base_url: str | None = None):
        from openai import OpenAI
        self.provider = provider
        self.model = model

        if provider == "vllm":
            api_key = os.getenv("VLLM_API_KEY", "dummy")
            self._client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key or api_key.startswith("your-"):
                raise ValueError("OPENAI_API_KEY bulunamadı.")
            self._client = OpenAI(api_key=api_key)

    def chat(self, messages, tools):
        return self._client.chat.completions.create(
            model=self.model, messages=messages, tools=tools,
            tool_choice="auto", temperature=0.3,
        )

    def simple_chat(self, messages):
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.3,
        )
        return resp.choices[0].message.content or ""


class _GeminiFunctionCaller:
    """Google Gemini function calling client."""

    def __init__(self, model: str):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key.startswith("your-"):
            raise ValueError("GEMINI_API_KEY bulunamadı.")
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model
        self.provider = "gemini"

        # Gemini tool definitions
        self._tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="vector_search",
                    description="Yerel vektör veritabanında anlamsal benzerlik araması yapar.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "query": types.Schema(type="STRING", description="Aranacak soru"),
                            "top_k": types.Schema(type="INTEGER", description="Maks sonuç"),
                        },
                        required=["query"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="web_search",
                    description="İnternette güncel bilgi arar.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "query": types.Schema(type="STRING", description="Aranacak konu"),
                        },
                        required=["query"],
                    ),
                ),
            ])
        ]

    def chat(self, messages, tools=None):
        """Gemini chat with function calling — returns a wrapper."""
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue  # handled via system_instruction
            elif role == "tool":
                # Gemini expects function response parts
                contents.append(self._types.Content(
                    role="function",
                    parts=[self._types.Part(function_response=self._types.FunctionResponse(
                        name=m.get("name", "unknown"),
                        response={"result": m["content"]},
                    ))],
                ))
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(self._types.Content(role=gemini_role, parts=[self._types.Part(text=m["content"])]))

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                tools=self._tools,
                system_instruction=_AGENT_SYSTEM,
                temperature=0.3,
            ),
        )
        return _GeminiResponseWrapper(resp)

    def simple_chat(self, messages):
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(self._types.Content(role=role, parts=[self._types.Part(text=m["content"])]))

        resp = self._client.models.generate_content(
            model=self.model, contents=contents,
            config=self._types.GenerateContentConfig(system_instruction=_AGENT_SYSTEM, temperature=0.3),
        )
        return resp.text or ""


class _GeminiResponseWrapper:
    """Wraps Gemini response to match OpenAI-like interface."""

    def __init__(self, resp):
        self._resp = resp
        candidate = resp.candidates[0] if resp.candidates else None
        self.choices = [_GeminiChoice(candidate)] if candidate else []


class _GeminiChoice:
    def __init__(self, candidate):
        self.message = _GeminiMessage(candidate)
        # Check if there are function calls
        has_fc = any(
            hasattr(p, "function_call") and p.function_call and p.function_call.name
            for p in (candidate.content.parts if candidate and candidate.content else [])
        )
        self.finish_reason = "tool_calls" if has_fc else "stop"


class _GeminiMessage:
    def __init__(self, candidate):
        self.tool_calls = []
        self.content = ""
        if not candidate or not candidate.content:
            return
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                self.tool_calls.append(_GeminiToolCall(part.function_call))
            elif hasattr(part, "text") and part.text:
                self.content += part.text


class _GeminiToolCall:
    def __init__(self, fc):
        self.id = fc.name
        self.function = _GeminiFunctionRef(fc)


class _GeminiFunctionRef:
    def __init__(self, fc):
        self.name = fc.name
        self.arguments = json.dumps(dict(fc.args) if fc.args else {})


class _ClaudeFunctionCaller:
    """Anthropic Claude function calling client."""

    def __init__(self, model: str):
        api_key = os.getenv("CLAUDE_API_KEY", "")
        if not api_key or api_key.startswith("your-"):
            raise ValueError("CLAUDE_API_KEY bulunamadı.")
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.provider = "claude"

        self._tools = [
            {
                "name": "vector_search",
                "description": "Yerel vektör veritabanında anlamsal benzerlik araması yapar.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Aranacak soru"},
                        "top_k": {"type": "integer", "description": "Maks sonuç", "default": 3},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "web_search",
                "description": "İnternette güncel bilgi arar.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Aranacak konu"},
                    },
                    "required": ["query"],
                },
            },
        ]

    def chat(self, messages, tools=None):
        claude_msgs = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model, max_tokens=1024, system=_AGENT_SYSTEM,
            messages=claude_msgs, tools=self._tools,
        )
        return _ClaudeResponseWrapper(resp)

    def simple_chat(self, messages):
        claude_msgs = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model, max_tokens=1024, system=_AGENT_SYSTEM,
            messages=claude_msgs,
        )
        return resp.content[0].text if resp.content else ""


class _ClaudeResponseWrapper:
    def __init__(self, resp):
        self.choices = [_ClaudeChoice(resp)]


class _ClaudeChoice:
    def __init__(self, resp):
        self.message = _ClaudeMessage(resp)
        has_tool = any(b.type == "tool_use" for b in resp.content)
        self.finish_reason = "tool_calls" if has_tool else "stop"


class _ClaudeMessage:
    def __init__(self, resp):
        self.tool_calls = []
        self.content = ""
        for block in resp.content:
            if block.type == "tool_use":
                self.tool_calls.append(_ClaudeToolCall(block))
            elif block.type == "text":
                self.content += block.text


class _ClaudeToolCall:
    def __init__(self, block):
        self.id = block.id
        self.function = _ClaudeFunctionRef(block)


class _ClaudeFunctionRef:
    def __init__(self, block):
        self.name = block.name
        self.arguments = json.dumps(block.input)


# ---------------------------------------------------------------------------
# AgenticRAGDemo
# ---------------------------------------------------------------------------


def _create_function_caller(provider: str, model: str):
    """Factory: provider'a göre function calling client oluşturur."""
    if provider == "openai":
        return _OpenAIFunctionCaller("openai", model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    elif provider == "gemini":
        return _GeminiFunctionCaller(model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    elif provider == "claude":
        return _ClaudeFunctionCaller(model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    elif provider == "vllm":
        return _OpenAIFunctionCaller(
            "vllm",
            model or os.getenv("VLLM_MODEL", ""),
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
        )
    elif provider == "ollama":
        # Ollama OpenAI-compatible endpoint
        return _OpenAIFunctionCaller(
            "ollama",
            model or os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Bilinmeyen provider: {provider}. Desteklenen: openai, gemini, claude, ollama, vllm")


class AgenticRAGDemo:
    """Çoklu LLM provider ile otonom araç seçimi yapan Agentic RAG demo."""

    MAX_ITERATIONS = 5

    def __init__(self, provider: str = "openai", model: str = "", collection_name: str = "rag_demo") -> None:
        self._caller = _create_function_caller(provider, model)
        self._provider = provider
        self._vector_tool = VectorSearchTool(collection_name=collection_name)
        self._web_tool = WebSearchTool()

        # OpenAI-format tool definitions (used by openai/vllm/ollama)
        self._tools = [VectorSearchTool.TOOL_DEFINITION, WebSearchTool.TOOL_DEFINITION]
        logger.info("AgenticRAGDemo hazır – provider=%s, model=%s", provider, self._caller.model)

    def ingest(self, data_dir: str) -> int:
        return self._vector_tool.ingest_directory(data_dir)

    def run(self, question: str) -> str:
        print(f"\n{_CYAN}{_BOLD}Soru:{_RESET} {question}")
        print(f"{_DIM}{'─' * 60}{_RESET}")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _AGENT_SYSTEM},
            {"role": "user", "content": question},
        ]

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n{_YELLOW}[İterasyon {iteration}/{self.MAX_ITERATIONS}]{_RESET}")

            response = self._caller.chat(messages, self._tools)
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "stop" or not msg.tool_calls:
                answer = msg.content or "Cevap üretildi."
                print(f"\n{_GREEN}{_BOLD}Nihai Cevap:{_RESET}")
                print(answer)
                return answer

            # Process tool calls
            # For OpenAI/vLLM: append assistant message
            if self._provider in ("openai", "vllm", "ollama"):
                messages.append(msg)

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"  {_MAGENTA}Araç:{_RESET} {tool_name}")
                print(f"  {_DIM}Parametreler: {tool_args}{_RESET}")

                result = self._execute_tool(tool_name, tool_args)
                print(f"  {_GREEN}Sonuç:{_RESET} {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")

                if self._provider in ("openai", "vllm", "ollama"):
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
                elif self._provider == "gemini":
                    messages.append({"role": "tool", "name": tool_name, "content": str(result)})
                elif self._provider == "claude":
                    messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": tc.id, "name": tool_name, "input": tool_args}]})
                    messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tc.id, "content": str(result)}]})

        print(f"\n{_YELLOW}Maksimum iterasyon ({self.MAX_ITERATIONS}) sınırına ulaşıldı.{_RESET}")
        answer = self._caller.simple_chat(
            messages + [{"role": "user", "content": "Toplanan bilgilere dayanarak soruyu yanıtla."}]
        )
        print(f"\n{_GREEN}{_BOLD}Nihai Cevap:{_RESET}")
        print(answer)
        return answer

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "vector_search":
            results = self._vector_tool.search(args.get("query", ""), args.get("top_k", 3))
            if not results:
                return "Vektör veritabanında ilgili bilgi bulunamadı."
            return "\n\n".join(f"[Kaynak {i}: {r.source} (skor: {r.score:.3f})]\n{r.text}" for i, r in enumerate(results, 1))
        elif tool_name == "web_search":
            return self._web_tool.search(args.get("query", ""))
        return f"Bilinmeyen araç: {tool_name}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_banner():
    print(f"""
{_BOLD}{'=' * 60}
  🤖 Agentic RAG Demo
  Çoklu LLM + Otonom Araç Seçimi
{'=' * 60}{_RESET}
""")


def main():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Agentic RAG Demo")
    parser.add_argument("--provider", choices=["openai", "gemini", "claude", "ollama", "vllm"],
                        default=os.getenv("LLM_PROVIDER", "openai"), help="LLM provider")
    parser.add_argument("--model", default=None, help="Model adı")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "examples" / "data"), help="Veri dizini")
    parser.add_argument("--collection", default="rag_demo", help="Koleksiyon adı")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maks iterasyon")
    args = parser.parse_args()

    _print_banner()

    try:
        print(f"{_CYAN}Demo başlatılıyor (provider={args.provider})...{_RESET}")
        demo = AgenticRAGDemo(provider=args.provider, model=args.model or "", collection_name=args.collection)
        AgenticRAGDemo.MAX_ITERATIONS = args.max_iterations
        print(f"{_GREEN}Demo hazır.{_RESET}\n")
    except (ValueError, ImportError, RuntimeError) as exc:
        print(f"{_RED}Hata: {exc}{_RESET}")
        sys.exit(1)

    data_path = Path(args.data_dir)
    if data_path.exists():
        print(f"{_CYAN}Veri yükleniyor: {args.data_dir}{_RESET}")
        try:
            n = demo.ingest(args.data_dir)
            print(f"{_GREEN}{n} chunk yüklendi.{_RESET}\n")
        except Exception as exc:
            print(f"{_YELLOW}Veri yükleme atlandı: {exc}{_RESET}\n")
    else:
        print(f"{_YELLOW}Veri dizini bulunamadı: {args.data_dir}{_RESET}\n")

    print(f"{_BOLD}Soru-Cevap (çıkmak için 'q'){_RESET}\n")

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
        print(f"\n{_YELLOW}Çıkılıyor...{_RESET}")

    print(f"{_DIM}Demo tamamlandı.{_RESET}")


if __name__ == "__main__":
    main()

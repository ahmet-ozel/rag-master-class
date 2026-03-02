"""
RAG Evaluation Script
=====================
RAG pipeline kalitesini Faithfulness, Answer Relevancy ve Context Precision
metrikleriyle ölçen değerlendirme scripti.

Kullanım:
    python evaluate.py --provider openai --qa-pairs qa_pairs.json
    python evaluate.py --provider ollama --model llama3.2 --output results.json

Metrikler (0-1 arası):
    - Faithfulness: Cevaptaki iddiaların bağlamda bulunup bulunmadığı
    - Answer Relevancy: Cevap ile soru arasındaki anlamsal benzerlik
    - Context Precision: Getirilen chunk'ların soruyla alakası
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup – allow importing from Classical-RAG
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
_CLASSICAL_RAG_DIR = _PROJECT_ROOT / "Classical-RAG"

if str(_CLASSICAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSICAL_RAG_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root
load_dotenv(_PROJECT_ROOT / ".env")

from demo_pipeline import ClassicalRAGPipeline, RAGResponse, ChunkResult

logger = logging.getLogger("rag_evaluator")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Complete evaluation report with per-metric scores."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    details: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, clamped to [0, 1]."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))


def _get_embedding_model():
    """Lazy-load sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _compute_faithfulness(
    answer: str,
    context: str,
    llm_generate,
) -> float:
    """Compute faithfulness by extracting claims and checking context support.

    Uses the LLM to:
    1. Extract factual claims from the answer.
    2. Check each claim against the retrieved context.

    Returns:
        Float between 0 and 1 (supported_claims / total_claims).
    """
    # Step 1 – extract claims
    extract_prompt = (
        "Extract all factual claims from the following answer. "
        "Return ONLY a JSON array of strings, one claim per element. "
        "If there are no factual claims, return an empty array [].\n\n"
        f"Answer: {answer}"
    )
    try:
        claims_raw = llm_generate(extract_prompt, "")
        # Try to parse JSON array from the response
        claims_raw = claims_raw.strip()
        # Handle markdown code blocks
        if claims_raw.startswith("```"):
            lines = claims_raw.split("\n")
            claims_raw = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )
        claims: List[str] = json.loads(claims_raw)
        if not isinstance(claims, list) or len(claims) == 0:
            return 1.0  # No claims to verify → faithful by default
    except (json.JSONDecodeError, Exception):
        logger.warning("Could not parse claims from LLM response; defaulting to 1.0")
        return 1.0

    # Step 2 – verify each claim
    supported = 0
    for claim in claims:
        verify_prompt = (
            "Given the following context and claim, determine if the claim "
            "is supported by the context. Answer ONLY 'yes' or 'no'.\n\n"
            f"Context: {context}\n\n"
            f"Claim: {claim}"
        )
        try:
            verdict = llm_generate(verify_prompt, "").strip().lower()
            if "yes" in verdict:
                supported += 1
        except Exception:
            logger.warning("LLM call failed for claim verification; skipping claim")

    score = supported / len(claims) if claims else 1.0
    return max(0.0, min(1.0, score))


def _compute_answer_relevancy(
    question: str,
    answer: str,
    embed_model,
) -> float:
    """Compute answer relevancy as cosine similarity between question and answer embeddings."""
    embeddings = embed_model.encode([question, answer])
    return _compute_cosine_similarity(embeddings[0], embeddings[1])


def _compute_context_precision(
    question: str,
    chunks: List[ChunkResult],
    embed_model,
) -> float:
    """Compute context precision as average cosine similarity between question and chunk embeddings."""
    if not chunks:
        return 0.0

    texts = [question] + [c.text for c in chunks]
    embeddings = embed_model.encode(texts)
    q_emb = embeddings[0]

    similarities = []
    for chunk_emb in embeddings[1:]:
        similarities.append(_compute_cosine_similarity(q_emb, chunk_emb))

    return float(np.mean(similarities)) if similarities else 0.0


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """RAG pipeline kalite değerlendirmesi.

    Faithfulness, Answer Relevancy ve Context Precision metriklerini hesaplar.
    """

    def __init__(self, pipeline: ClassicalRAGPipeline) -> None:
        """Initialize evaluator with a RAG pipeline instance.

        Args:
            pipeline: A ClassicalRAGPipeline that has already ingested documents.
        """
        self._pipeline = pipeline
        self._embed_model = _get_embedding_model()

    def evaluate(self, qa_pairs_path: str) -> EvaluationReport:
        """Run evaluation over QA pairs and return an EvaluationReport.

        Args:
            qa_pairs_path: Path to a JSON file with ``qa_pairs`` array.

        Returns:
            EvaluationReport with aggregate and per-question metrics.
        """
        qa_path = Path(qa_pairs_path)
        if not qa_path.exists():
            raise FileNotFoundError(
                f"QA pairs file not found: {qa_pairs_path}\n"
                "Expected format: {\"qa_pairs\": [{\"question\": ..., \"ground_truth\": ...}]}"
            )

        with open(qa_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {qa_pairs_path}: {exc}\n"
                    "Expected format: {\"qa_pairs\": [{\"question\": ..., \"ground_truth\": ...}]}"
                ) from exc

        qa_pairs = data.get("qa_pairs", [])
        if not qa_pairs:
            raise ValueError(f"No qa_pairs found in {qa_pairs_path}")

        details: List[Dict[str, Any]] = []
        faith_scores: List[float] = []
        relevancy_scores: List[float] = []
        precision_scores: List[float] = []

        total = len(qa_pairs)
        for idx, pair in enumerate(qa_pairs, 1):
            question = pair["question"]
            print(f"\n[{idx}/{total}] Evaluating: {question}")

            try:
                response: RAGResponse = self._pipeline.query(question)
            except Exception as exc:
                logger.warning("Pipeline query failed for '%s': %s", question, exc)
                details.append({
                    "question": question,
                    "error": str(exc),
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                })
                faith_scores.append(0.0)
                relevancy_scores.append(0.0)
                precision_scores.append(0.0)
                continue

            # Build context string from sources
            context = "\n\n".join(c.text for c in response.sources)

            # --- Faithfulness ---
            faith = _compute_faithfulness(
                answer=response.answer,
                context=context,
                llm_generate=self._pipeline._llm.generate,
            )
            faith_scores.append(faith)

            # --- Answer Relevancy ---
            relevancy = _compute_answer_relevancy(
                question=question,
                answer=response.answer,
                embed_model=self._embed_model,
            )
            relevancy_scores.append(relevancy)

            # --- Context Precision ---
            precision = _compute_context_precision(
                question=question,
                chunks=response.sources,
                embed_model=self._embed_model,
            )
            precision_scores.append(precision)

            detail = {
                "question": question,
                "answer": response.answer,
                "ground_truth": pair.get("ground_truth", ""),
                "faithfulness": round(faith, 4),
                "answer_relevancy": round(relevancy, 4),
                "context_precision": round(precision, 4),
                "num_sources": len(response.sources),
            }
            details.append(detail)

            print(
                f"  Faithfulness: {faith:.4f} | "
                f"Relevancy: {relevancy:.4f} | "
                f"Precision: {precision:.4f}"
            )

        avg_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
        avg_relevancy = float(np.mean(relevancy_scores)) if relevancy_scores else 0.0
        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0

        report = EvaluationReport(
            faithfulness=round(max(0.0, min(1.0, avg_faith)), 4),
            answer_relevancy=round(max(0.0, min(1.0, avg_relevancy)), 4),
            context_precision=round(max(0.0, min(1.0, avg_precision)), 4),
            details=details,
        )

        self._print_report(report)
        return report

    def export_json(self, report: EvaluationReport, output_path: str) -> None:
        """Export evaluation report to a JSON file.

        Args:
            report: The EvaluationReport to export.
            output_path: Destination file path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(report)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Results exported to {output_path}")

    @staticmethod
    def _print_report(report: EvaluationReport) -> None:
        """Print a formatted summary table to the terminal."""
        print("\n" + "=" * 60)
        print("  RAG Evaluation Report")
        print("=" * 60)
        print(f"  Timestamp       : {report.timestamp}")
        print(f"  Questions       : {len(report.details)}")
        print("-" * 60)
        print(f"  Faithfulness    : {report.faithfulness:.4f}")
        print(f"  Answer Relevancy: {report.answer_relevancy:.4f}")
        print(f"  Context Precision: {report.context_precision:.4f}")
        print("=" * 60)

        if report.details:
            print("\nPer-question breakdown:")
            print(f"  {'#':<4} {'Faith':>7} {'Relev':>7} {'Prec':>7}  Question")
            print(f"  {'─'*4} {'─'*7} {'─'*7} {'─'*7}  {'─'*30}")
            for i, d in enumerate(report.details, 1):
                f_val = d.get("faithfulness", 0.0)
                r_val = d.get("answer_relevancy", 0.0)
                p_val = d.get("context_precision", 0.0)
                q_short = d["question"][:30]
                print(f"  {i:<4} {f_val:>7.4f} {r_val:>7.4f} {p_val:>7.4f}  {q_short}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run evaluation from the command line."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --provider openai\n"
            "  python evaluate.py --provider ollama --model llama3.2\n"
            "  python evaluate.py --provider openai --qa-pairs custom_qa.json --output results.json\n"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: provider-specific default)",
    )
    parser.add_argument(
        "--qa-pairs",
        default=str(_EVAL_DIR / "qa_pairs.json"),
        help="Path to QA pairs JSON file (default: evaluation/qa_pairs.json)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_PROJECT_ROOT / "examples" / "data"),
        help="Path to sample data directory for ingestion",
    )
    parser.add_argument(
        "--output",
        default=str(_EVAL_DIR / "results.json"),
        help="Output path for JSON results (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  RAG Pipeline Evaluation")
    print("=" * 60)
    print(f"  Provider  : {args.provider}")
    print(f"  Model     : {args.model or 'default'}")
    print(f"  QA Pairs  : {args.qa_pairs}")
    print(f"  Data Dir  : {args.data_dir}")
    print(f"  Output    : {args.output}")
    print("=" * 60)

    # 1. Initialize pipeline
    print("\n📦 Initializing pipeline...")
    pipeline = ClassicalRAGPipeline(
        llm_provider=args.provider,
        model_name=args.model,
    )

    # 2. Ingest sample data
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        files = [
            str(f)
            for f in data_dir.iterdir()
            if f.is_file() and f.suffix in (".txt", ".csv", ".pdf")
        ]
        if files:
            print(f"\n📄 Ingesting {len(files)} file(s)...")
            count = pipeline.ingest(files)
            print(f"   ✅ {count} chunks ingested")
        else:
            print(f"\n⚠️  No supported files found in {data_dir}")
    else:
        print(f"\n⚠️  Data directory not found: {data_dir}")

    # 3. Run evaluation
    print("\n🔍 Running evaluation...")
    evaluator = RAGEvaluator(pipeline)
    report = evaluator.evaluate(args.qa_pairs)

    # 4. Export results
    evaluator.export_json(report, args.output)


if __name__ == "__main__":
    main()

"""
Document Chunking and Embedding Application with Enhanced CSV/Excel Support
Sentence-aware chunking that never breaks words mid-token.
"""
import io
import json
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import os
import zipfile

# Optional AES-encrypted ZIP support
try:
    import pyzipper  # type: ignore
except Exception:
    pyzipper = None

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODELS, APP_CONFIG, get_model_config


# ============= UTILS =============
import logging
from datetime import datetime

import pdfplumber
from docx import Document


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def log_exception(e: Exception, context: str = ""):
    """Log exception with context."""
    logger = setup_logging()
    logger.error(f"{context}{str(e)}", exc_info=True)


# ---------- Whitespace normalisation ----------
_WS_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_MULTI_SPACE_RE = re.compile(r"[ \t\f\r]+")
_NEWLINE_SPACE_RE = re.compile(r"[ \t\f\r]*\n[ \t\f\r]*")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: NBSP → space, remove zero-width chars, collapse runs."""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    if not text:
        return ""
    text = text.replace("\u00A0", " ").replace("\u202F", " ")
    text = _WS_ZW_RE.sub("", text)
    text = _NEWLINE_SPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------- Token helpers ----------

def get_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    """Return the number of tokens (no special tokens)."""
    try:
        return int(len(tokenizer.encode(text, add_special_tokens=False)))
    except Exception:
        return int(max(1, len(text.split())))


# ---------- Sentence boundary detection ----------

def find_sentence_boundaries(text: str) -> List[int]:
    """
    Find reliable sentence-end positions in *text*.
    Respects common abbreviations and numbers to avoid false splits.
    """
    boundaries: List[int] = []
    text_len = len(text)

    # Common abbreviations (multilingual) – period after these is NOT a sentence end
    abbreviations = {
        # Academic / titles
        "dr", "prof", "doc", "sr", "jr", "mr", "mrs", "ms", "phd", "md",
        # General
        "vs", "etc", "inc", "ltd", "corp", "co", "dept", "est", "approx",
        # References / pages
        "pg", "pp", "vol", "fig", "eq", "ref", "no",
        # Address
        "st", "ave", "blvd", "apt", "rd", "fl",
        # Turkish extras (kept for multilingual support)
        "yrd", "gör", "bkz", "vb", "a.ş", "şti", "tic", "san",
    }

    i = 0
    while i < text_len:
        char = text[i]

        if char == ".":
            # Collect the word immediately before the period
            word_before = ""
            j = i - 1
            while j >= 0 and text[j] not in " \n\t":
                word_before = text[j] + word_before
                j -= 1
            word_lower = word_before.lower().rstrip(".")

            # Skip if the character before the period is a digit (e.g. "100.")
            if word_before and word_before[-1].isdigit():
                i += 1
                continue

            # Skip known abbreviations
            if word_lower in abbreviations:
                i += 1
                continue

            if i + 1 < text_len:
                nxt = text[i + 1]
                if nxt in " \n\t":
                    k = i + 2
                    while k < text_len and text[k] in " \n\t":
                        k += 1
                    if k < text_len and text[k].isupper():
                        boundaries.append(i + 1)
                elif nxt == "\n":
                    boundaries.append(i + 1)
            elif i == text_len - 1:
                boundaries.append(i + 1)

        elif char in "!?":
            if i + 1 < text_len and text[i + 1] in " \n\t":
                boundaries.append(i + 1)
            elif i == text_len - 1:
                boundaries.append(i + 1)

        i += 1

    return boundaries


# ---------- Smart chunking ----------

def smart_chunk_with_sentences(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Token-based chunking that respects sentence boundaries.
    Falls back to word boundaries when no sentence end is found.
    Overlap is applied while **guaranteeing forward progress** (no infinite loops).
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    text = text.strip()
    chunks: List[str] = []

    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(full_tokens) <= max_tokens:
        return [text]

    # Build sorted set of sentence boundaries (incl. start / end)
    sentence_boundaries = find_sentence_boundaries(text)
    if 0 not in sentence_boundaries:
        sentence_boundaries.append(0)
    if len(text) not in sentence_boundaries:
        sentence_boundaries.append(len(text))
    sentence_boundaries = sorted(set(sentence_boundaries))

    char_per_token = max(1e-6, len(text) / max(1, len(full_tokens)))

    start_pos = 0
    safety = 0
    max_iters = 100_000

    while start_pos < len(text) and safety < max_iters:
        safety += 1

        approx_chars = int(max_tokens * char_per_token)
        target_end = min(len(text), start_pos + max(1, approx_chars))

        # Find the best sentence boundary within [start_pos, target_end]
        best: Optional[int] = None
        for b in sentence_boundaries:
            if start_pos < b <= target_end:
                best = b

        if best is None:
            # Fall back to word boundary
            b = target_end
            while b > start_pos and b < len(text) and not text[b].isspace():
                b -= 1
            best = target_end if b == start_pos else b

        chunk_text = text[start_pos:best].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if best >= len(text):
            break

        # Overlap with forward-progress guarantee
        if overlap_tokens > 0:
            overlap_chars = int(overlap_tokens * char_per_token)
            next_start = max(0, best - overlap_chars)
            while next_start < len(text) and text[next_start] in " \n\t":
                next_start += 1
            if next_start <= start_pos:
                next_start = best
        else:
            next_start = best

        if next_start <= start_pos:
            next_start = min(len(text), start_pos + 1)

        start_pos = next_start

    return chunks


def sentence_chunk_with_overlap(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """Convenience wrapper for smart_chunk_with_sentences."""
    return smart_chunk_with_sentences(text, tokenizer, max_tokens, overlap_tokens)


# ---------- File readers ----------

def read_pdf_content(file) -> str:
    """Extract text from a PDF file."""
    try:
        try:
            file.seek(0)
        except Exception:
            pass
        parts: List[str] = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return normalize_whitespace("\n".join(parts))
    except Exception as e:
        log_exception(e, "PDF reading error: ")
        return ""


def read_docx_content(file) -> str:
    """Extract text from a DOCX file."""
    try:
        try:
            file.seek(0)
        except Exception:
            pass
        doc = Document(file)
        return normalize_whitespace("\n".join(p.text for p in doc.paragraphs))
    except Exception as e:
        log_exception(e, "DOCX reading error: ")
        return ""


def read_txt_content(file) -> str:
    """Read a plain-text file with robust encoding fallbacks."""
    encodings = [
        "utf-8", "utf-8-sig", "windows-1254", "iso-8859-9", "latin1", "iso-8859-1",
    ]
    for enc in encodings:
        try:
            try:
                file.seek(0)
            except Exception:
                pass
            data = file.read()
            if isinstance(data, str):
                return normalize_whitespace(data)
            return normalize_whitespace(data.decode(enc))
        except Exception:
            continue
    try:
        try:
            file.seek(0)
        except Exception:
            pass
        raw = file.read()
        if isinstance(raw, str):
            return normalize_whitespace(raw)
        return normalize_whitespace(raw.decode("utf-8", errors="ignore"))
    except Exception as e:
        log_exception(e, "TXT reading error: ")
        return ""


def read_tabular_data(file, file_extension: str) -> Optional[pd.DataFrame]:
    """Read tabular data from CSV or Excel."""
    try:
        if file_extension == "csv":
            for enc in ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "windows-1254"]:
                try:
                    file.seek(0)
                    return pd.read_csv(file, encoding=enc)
                except Exception:
                    continue
        elif file_extension in ("xlsx", "xls"):
            file.seek(0)
            engine = "openpyxl" if file_extension == "xlsx" else "xlrd"
            return pd.read_excel(file, engine=engine)
    except Exception as e:
        log_exception(e, f"Error reading {file_extension}: ")
    return None


# ---------- Table helpers ----------

def is_structured_table(df: pd.DataFrame, min_cols: int = 2, min_rows: int = 2) -> bool:
    if df is None or df.empty:
        return False
    return len(df.columns) >= min_cols and len(df) >= min_rows


def dataframe_to_text(df: pd.DataFrame) -> str:
    """Flatten a DataFrame into pipe-separated text."""
    lines: List[str] = []
    lines.append(" | ".join(normalize_whitespace(str(c)) for c in df.columns))
    for _, row in df.iterrows():
        lines.append(
            " | ".join(normalize_whitespace(str(v)) if pd.notna(v) else "" for v in row)
        )
    return normalize_whitespace("\n".join(lines))


def create_chunk_record(
    file_name: str,
    source_type: str,
    text: str,
    part_index: int,
    row_index: Optional[int] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a standardized chunk record dict."""
    chunk_id = f"{file_name}_{row_index if row_index is not None else 'doc'}_{part_index}"
    record: Dict[str, Any] = {
        "file": file_name,
        "source": source_type,
        "text": text,
        "part": part_index,
        "chunk_id": chunk_id,
        "timestamp": datetime.now().isoformat(),
    }
    if row_index is not None:
        record["row"] = row_index
    if metadata:
        record.update(metadata)
    return record


def row_to_text_advanced(
    row: pd.Series,
    include_column_names: bool,
    use_abbreviations: bool,
    add_column_prefix: bool,
    values_only_threshold: int,
    abbreviations: Optional[Dict[str, str]] = None,
) -> str:
    """Convert a single DataFrame row to a text string."""
    parts: List[str] = []
    non_empty = sum(1 for v in row if pd.notna(v) and str(v).strip())

    if non_empty <= values_only_threshold and not include_column_names:
        for v in row:
            if pd.notna(v) and str(v).strip():
                parts.append(normalize_whitespace(str(v)))
    else:
        for col, v in row.items():
            if pd.notna(v) and str(v).strip():
                val = normalize_whitespace(str(v))
                if include_column_names or add_column_prefix:
                    if use_abbreviations and abbreviations:
                        prefix = abbreviations.get(str(col), str(col))
                    else:
                        prefix = str(col)
                    parts.append(f"{normalize_whitespace(prefix)}: {val}")
                else:
                    parts.append(val)

    return normalize_whitespace(" | ".join(parts))


# ---------- ZIP archive support ----------

SUPPORTED_IN_ZIP = {"pdf", "docx", "txt", "csv", "xlsx", "xls"}


def process_zip_archive(
    uploaded_zip,
    processor,  # DocumentProcessor
    include_column_names: bool,
    values_only_threshold: int,
    min_cols_for_table: int,
    min_rows_for_table: int,
    attach_row_data: bool,
    flatten_row_values_to_root: bool,
    password: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract supported files from a ZIP archive and process them."""
    chunks: List[Dict[str, Any]] = []
    pwd_bytes = password.encode("utf-8") if isinstance(password, str) and password else None
    zip_label = getattr(uploaded_zip, "name", "ZIP")

    def _process_inner(name: str, data: bytes) -> List[Dict[str, Any]]:
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in SUPPORTED_IN_ZIP:
            return []
        bio = io.BytesIO(data)
        setattr(bio, "name", os.path.basename(name))
        if ext in ("pdf", "docx", "txt"):
            return processor.process_text_document(bio, ext)
        return processor.process_tabular_document(
            file=bio,
            file_extension=ext,
            include_column_names=include_column_names,
            values_only_threshold=int(values_only_threshold),
            min_cols_for_table=int(min_cols_for_table),
            min_rows_for_table=int(min_rows_for_table),
            attach_row_data=bool(attach_row_data),
            flatten_row_values_to_root=bool(flatten_row_values_to_root),
        )

    # --- Standard zipfile (ZipCrypto) ---
    try:
        try:
            uploaded_zip.seek(0)
        except Exception:
            pass

        with zipfile.ZipFile(uploaded_zip) as zf:
            if pwd_bytes:
                try:
                    zf.setpassword(pwd_bytes)
                except Exception:
                    pass

            members = [m for m in zf.infolist() if not m.is_dir()]
            processed = 0

            for info in members:
                name = info.filename
                if name.startswith("__MACOSX") or name.endswith(".DS_Store"):
                    continue
                try:
                    data = zf.read(info, pwd=pwd_bytes)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "password" in msg or "encrypted" in msg or "bad password" in msg:
                        raise
                    raise

                chunks.extend(_process_inner(name, data))
                processed += 1

            if processed > 0:
                st.success(f"Processed {processed} file(s) from {zip_label} → {len(chunks)} chunks.")
                return chunks

    except Exception as e_std:
        if pyzipper is None:
            if pwd_bytes is None:
                st.error(f"{zip_label}: Password required. Please enter the ZIP password.")
            else:
                st.error(
                    f"{zip_label}: Cannot read ZIP. Wrong password or AES encryption. "
                    "Install `pyzipper` to support AES-encrypted ZIPs."
                )
            log_exception(e_std, f"ZIP read failed {zip_label}: ")
            return chunks

        # --- pyzipper fallback (AES) ---
        try:
            try:
                uploaded_zip.seek(0)
            except Exception:
                pass

            with pyzipper.AESZipFile(uploaded_zip) as zf2:  # type: ignore[union-attr]
                if pwd_bytes:
                    zf2.pwd = pwd_bytes
                else:
                    st.error(f"{zip_label}: Password required (AES-encrypted ZIP).")
                    return chunks

                members = [m for m in zf2.infolist() if not m.is_dir()]
                processed = 0

                for info in members:
                    name = info.filename
                    if name.startswith("__MACOSX") or name.endswith(".DS_Store"):
                        continue
                    try:
                        data = zf2.read(info)
                    except Exception as e_read:
                        st.error(f"{zip_label}/{name}: Wrong password or file unreadable.")
                        log_exception(e_read, f"ZIP AES read failed {zip_label}/{name}: ")
                        continue

                    chunks.extend(_process_inner(name, data))
                    processed += 1

                if processed == 0:
                    st.warning(f"No supported files found inside {zip_label}.")
                else:
                    st.success(
                        f"Processed {processed} file(s) from {zip_label} (AES) → {len(chunks)} chunks."
                    )

        except Exception as e_aes:
            st.error(f"{zip_label}: Cannot process ZIP. Password or encryption format not supported.")
            log_exception(e_aes, f"ZIP AES fallback failed {zip_label}: ")

    return chunks


# ============= APPLICATION =============

def get_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource(show_spinner="🔄 Loading model…")
def _load_model(model_name: str, device: str) -> Tuple[AutoTokenizer, SentenceTransformer]:
    """Load tokenizer + SentenceTransformer (cached across reruns)."""
    cfg = get_model_config(model_name)
    model_id = cfg["model_id"]
    logger = setup_logging()
    logger.info(f"Loading model: {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = SentenceTransformer(model_id, device=device)
    logger.info(f"Model loaded: {model_id}")
    return tokenizer, model


class DocumentProcessor:
    """Process text and tabular documents into embedding-ready chunks."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_tokens: int,
        overlap: int,
        output_text_column: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.output_text_column = output_text_column
        self.logger = setup_logging()

    # ----- Text documents -----

    def process_text_document(self, file, file_extension: str) -> List[Dict[str, Any]]:
        """Process PDF / DOCX / TXT with sentence-aware chunking."""
        chunks: List[Dict[str, Any]] = []
        fname = getattr(file, "name", "unknown")

        try:
            if file_extension == "pdf":
                content = read_pdf_content(file)
            elif file_extension == "docx":
                content = read_docx_content(file)
            elif file_extension == "txt":
                content = read_txt_content(file)
            else:
                self.logger.warning(f"Unsupported text format: {file_extension}")
                return chunks

            if not content or not content.strip():
                st.warning(f"{fname}: Empty or no text could be extracted")
                return chunks

            chunk_texts = sentence_chunk_with_overlap(
                content, self.tokenizer, self.max_tokens, self.overlap
            )

            for i, ct in enumerate(chunk_texts):
                rec = create_chunk_record(
                    file_name=fname,
                    source_type=file_extension,
                    text=ct,
                    part_index=i,
                    metadata={"original_length": len(content), "sentence_aware": True},
                )
                if self.output_text_column != "text":
                    rec[self.output_text_column] = rec.pop("text")
                rec["token_count"] = get_token_count(ct, self.tokenizer)
                chunks.append(rec)

            st.success(f"✅ {fname} → {len(chunks)} sentence-aware chunks")

        except Exception as e:
            log_exception(e, f"Failed to process {fname}: ")
            st.error(f"Error processing {fname}: {e}")

        return chunks

    # ----- Tabular documents -----

    def process_tabular_document(
        self,
        file,
        file_extension: str,
        include_column_names: bool,
        values_only_threshold: int,
        min_cols_for_table: int,
        min_rows_for_table: int,
        attach_row_data: bool,
        flatten_row_values_to_root: bool,
    ) -> List[Dict[str, Any]]:
        """Process CSV / XLSX / XLS – each row becomes its own chunk."""
        chunks: List[Dict[str, Any]] = []
        fname = getattr(file, "name", "unknown")

        try:
            df = read_tabular_data(file, file_extension)
            if df is None or df.empty:
                st.warning(f"{fname}: No tabular data or file is empty")
                return chunks

            all_columns = [str(c) for c in df.columns]
            structured = is_structured_table(df, min_cols=min_cols_for_table, min_rows=min_rows_for_table)

            if structured:
                part_idx = 0
                rows_done = 0

                for ridx, row in df.iterrows():
                    text = row_to_text_advanced(
                        row=row,
                        include_column_names=include_column_names,
                        use_abbreviations=False,
                        add_column_prefix=False,
                        values_only_threshold=int(values_only_threshold),
                        abbreviations=None,
                    )
                    if not text.strip():
                        continue

                    tok_count = get_token_count(text, self.tokenizer)

                    if tok_count <= self.max_tokens:
                        meta: Dict[str, Any] = {
                            "is_tabular": True,
                            "table_type": "structured",
                            "total_rows": len(df),
                            "total_columns": len(all_columns),
                            "table_columns": all_columns,
                            "row_as_chunk": True,
                            "single_row": True,
                        }
                        if attach_row_data:
                            meta["row_data"] = self._row_dict(row)

                        rec = create_chunk_record(
                            file_name=fname,
                            source_type=file_extension,
                            text=text,
                            part_index=part_idx,
                            row_index=int(ridx),
                            metadata=meta,
                        )
                        if self.output_text_column != "text":
                            rec[self.output_text_column] = rec.pop("text")
                        rec["token_count"] = int(tok_count)
                        if flatten_row_values_to_root:
                            self._flatten_row(rec, meta)
                        chunks.append(rec)
                        rows_done += 1
                        part_idx += 1
                    else:
                        # Row too long – split into sub-chunks
                        rd = self._row_dict(row) if attach_row_data else {}
                        sub_texts = self._split_long_text(text)
                        rows_done += 1

                        for si, st_text in enumerate(sub_texts):
                            meta = {
                                "is_tabular": True,
                                "table_type": "structured",
                                "total_rows": len(df),
                                "total_columns": len(all_columns),
                                "table_columns": all_columns,
                                "row_split": True,
                                "original_row_index": int(ridx),
                                "row_split_index": si,
                                "row_split_total": len(sub_texts),
                            }
                            if attach_row_data:
                                meta["row_data"] = rd

                            rec = create_chunk_record(
                                file_name=fname,
                                source_type=file_extension,
                                text=st_text,
                                part_index=part_idx,
                                row_index=int(ridx),
                                metadata=meta,
                            )
                            if self.output_text_column != "text":
                                rec[self.output_text_column] = rec.pop("text")
                            rec["token_count"] = int(get_token_count(st_text, self.tokenizer))
                            if flatten_row_values_to_root:
                                self._flatten_row(rec, meta)
                            chunks.append(rec)
                            part_idx += 1

                st.success(f"✅ {fname} → {len(chunks)} row-level chunks (structured table)")

            else:
                # Unstructured table – flatten to prose and chunk normally
                flat = dataframe_to_text(df)
                if flat.strip():
                    for i, ct in enumerate(
                        smart_chunk_with_sentences(flat, self.tokenizer, self.max_tokens, self.overlap)
                    ):
                        meta = {
                            "is_tabular": True,
                            "table_type": "unstructured",
                            "flattened_table": True,
                            "total_rows": len(df),
                            "total_columns": len(all_columns),
                            "table_columns": all_columns,
                        }
                        rec = create_chunk_record(
                            file_name=fname,
                            source_type=file_extension,
                            text=ct,
                            part_index=i,
                            metadata=meta,
                        )
                        if self.output_text_column != "text":
                            rec[self.output_text_column] = rec.pop("text")
                        rec["token_count"] = int(get_token_count(ct, self.tokenizer))
                        chunks.append(rec)

                st.info(f"ℹ️ {fname} → {len(chunks)} document-style chunks (unstructured table)")

        except Exception as e:
            log_exception(e, f"Table processing error {fname}: ")
            st.error(f"Error processing table {fname}: {e}")

        return chunks

    # ----- Internal helpers -----

    def _row_dict(self, row: pd.Series) -> Dict[str, str]:
        """Convert a row to {col: value} dict."""
        d: Dict[str, str] = {}
        try:
            for k, v in row.items():
                d[str(k)] = "" if pd.isna(v) else str(v).strip()
        except Exception:
            pass
        return d

    def _split_long_text(self, text: str) -> List[str]:
        """Split text that exceeds max_tokens into word-aligned parts."""
        words = text.split()
        parts: List[str] = []
        buf: List[str] = []
        buf_tok = 0

        for w in words:
            wt = get_token_count(w, self.tokenizer)
            cost = wt + (1 if buf else 0)
            if buf_tok + cost <= self.max_tokens:
                buf.append(w)
                buf_tok += cost
            else:
                if buf:
                    parts.append(" ".join(buf))
                buf = [w]
                buf_tok = wt
        if buf:
            parts.append(" ".join(buf))
        return parts

    @staticmethod
    def _flatten_row(rec: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """Copy row_data values to root level of the record."""
        reserved = {"file", "source", "text", "part", "row", "chunk_id", "token_count", "timestamp"}
        try:
            for k, v in (meta.get("row_data") or {}).items():
                key = str(k)
                if key and key not in reserved and isinstance(v, str) and v.strip():
                    rec.setdefault(key, v)
        except Exception:
            pass


# ============= UI =============

def main():
    """Main Streamlit application."""

    st.set_page_config(page_title="Document Chunking & Embedding", page_icon="📄", layout="wide")
    st.title(APP_CONFIG["title"])
    st.markdown("---")

    st.info("✨ **Features:**")
    st.markdown(
        """
- 📝 **Sentence-Aware:** Chunks never split mid-sentence
- 🔤 **Word Integrity:** Words are never cut in half
- 📊 **CSV/Excel:** Each row becomes a separate chunk
- 🔠 **Text Normalization:** Lowercase / uppercase conversion
- 📦 **ZIP Download:** JSON + NPY bundled together
        """
    )

    # --- Model selection ---
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        model_name = st.selectbox(
            "🧠 Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            help="Choose a model based on language and quality needs",
        )
        mcfg = get_model_config(model_name)
        st.info(f"📝 {mcfg.get('description', 'No description')}")

    with col2:
        st.markdown("**Model Info:**")
        st.markdown(f"- Max Tokens: {mcfg.get('max_length', 'N/A')}")
        st.markdown(f"- Language: {mcfg.get('language', 'N/A')}")

    with col3:
        device = get_device()
        st.markdown("**Device:**")
        st.markdown(f"- {device.upper()}")
        if device == "cuda":
            try:
                st.markdown(f"- GPUs: {torch.cuda.device_count()}")
                st.markdown(f"- GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

    # --- Load model ---
    try:
        tokenizer, model = _load_model(model_name, device)
        model_max_len = mcfg.get("max_length", 512)
    except Exception:
        st.error("Failed to load model. Please try again.")
        st.stop()

    # --- Settings ---
    st.subheader("⚙️ Processing Settings")
    st.markdown("### 📝 General")

    dv = APP_CONFIG["default_values"]

    cA, cB, cC, cD = st.columns(4)
    with cA:
        max_tokens = st.slider(
            "Chunk max tokens",
            64, int(model_max_len),
            min(int(dv["max_tokens"]), int(model_max_len)),
            step=16,
            help="Maximum tokens per chunk (respects sentence boundaries)",
        )
    with cB:
        overlap = st.slider(
            "Chunk overlap (tokens)",
            0, min(256, int(model_max_len) // 2),
            min(int(dv["overlap"]), min(256, int(model_max_len) // 2)),
            step=8,
            help="Overlap between consecutive chunks",
        )
    with cC:
        values_only_threshold = st.number_input(
            "Values-only threshold", 0, 10, int(dv["values_only_threshold"]),
            help="If a row has ≤ this many non-empty cells, write values only",
        )
    with cD:
        min_cols_for_table = st.number_input(
            "Min columns (table)", 1, 20, int(dv["min_cols_for_table"]),
            help="Minimum columns to treat as structured table",
        )

    # Tabular
    st.markdown("### 📊 Table / CSV / Excel Settings")
    st.warning("⚠️ **Note:** In CSV/Excel files each row becomes a separate chunk. Rows are never merged!")

    cE, cF, cG = st.columns(3)
    with cE:
        include_column_names = st.checkbox(
            "Include column names in text",
            bool(dv["include_column_names"]),
            help="Format row text as 'Column: Value'",
        )
    with cF:
        min_rows_for_table = st.number_input(
            "Min rows (table)", 1, 100, int(dv["min_rows_for_table"]),
        )
    with cG:
        st.info("CSV/Excel: 1 row = 1 chunk")

    cI, cJ, cK = st.columns(3)
    with cI:
        attach_row_data = st.checkbox(
            "Attach row data to JSON",
            bool(dv.get("attach_row_data", True)),
            help="Embed full column→value mapping in each chunk",
        )
    with cJ:
        st.info("✅ Sentences are never split!")
    with cK:
        flatten_row_values_to_root = st.checkbox(
            "Flatten column values to root JSON",
            bool(dv.get("flatten_row_values_to_root", True)),
            help='E.g. adds "Name": "Alice" at root level for CSV rows',
        )

    # Text normalization
    st.markdown("### 🔤 Text Normalization")
    st.info("ℹ️ Normalization is applied **after** chunks are created")

    n1, n2, n3 = st.columns(3)
    with n1:
        to_lower = st.checkbox("🔡 Convert to lowercase", False)
    with n2:
        to_upper = st.checkbox("🔠 Convert to UPPERCASE", False)
    with n3:
        if to_lower and to_upper:
            st.warning("⚠️ Both selected – UPPERCASE takes priority.")
        else:
            st.success("✅ Ready")

    # Output
    st.markdown("### 💾 Output Settings")
    output_text_column = st.text_input(
        "Text column name", "text",
        help="Key name for the text field in JSON (e.g. text, chunk_text, content)",
    )

    st.markdown("---")

    # --- File upload ---
    allowed = list(APP_CONFIG["supported_formats"]) + ["zip"]
    uploaded_files = st.file_uploader(
        f"📁 Upload Documents ({', '.join(e.upper() for e in allowed)})",
        type=allowed,
        accept_multiple_files=True,
        key="doc_uploader",
        help=f"Up to {APP_CONFIG['max_files']} files",
    )

    if not uploaded_files and not st.session_state.get("results_ready"):
        st.info("👆 Upload files to get started")
        return

    if uploaded_files and len(uploaded_files) > APP_CONFIG["max_files"]:
        st.error(f"Maximum {APP_CONFIG['max_files']} files allowed!")
        return

    # ZIP passwords
    zips = [f for f in (uploaded_files or []) if f.name.lower().endswith(".zip")]
    if zips:
        with st.expander("🔐 ZIP Passwords", expanded=True):
            st.info("Enter passwords for encrypted ZIP files.")
            for z in zips:
                st.text_input(f"{z.name} password", type="password", key=f"zip_pwd::{z.name}")

    # --- Process ---
    if uploaded_files and st.button("🚀 Process & Generate Embeddings", type="primary"):
        all_chunks: List[Dict[str, Any]] = []
        proc = DocumentProcessor(tokenizer, max_tokens, overlap, output_text_column)

        prog = st.progress(0)
        status = st.empty()
        total = len(uploaded_files)

        for idx, file in enumerate(uploaded_files, 1):
            name = file.name
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            status.info(f"Processing ({idx}/{total}): {name}")
            cur: List[Dict[str, Any]] = []

            if ext == "zip":
                cur = process_zip_archive(
                    uploaded_zip=file,
                    processor=proc,
                    include_column_names=include_column_names,
                    values_only_threshold=int(values_only_threshold),
                    min_cols_for_table=int(min_cols_for_table),
                    min_rows_for_table=int(min_rows_for_table),
                    attach_row_data=bool(attach_row_data),
                    flatten_row_values_to_root=bool(flatten_row_values_to_root),
                    password=st.session_state.get(f"zip_pwd::{name}"),
                )
            elif ext in ("pdf", "docx", "txt"):
                cur = proc.process_text_document(file, ext)
            elif ext in ("csv", "xlsx", "xls"):
                cur = proc.process_tabular_document(
                    file=file,
                    file_extension=ext,
                    include_column_names=include_column_names,
                    values_only_threshold=int(values_only_threshold),
                    min_cols_for_table=int(min_cols_for_table),
                    min_rows_for_table=int(min_rows_for_table),
                    attach_row_data=bool(attach_row_data),
                    flatten_row_values_to_root=bool(flatten_row_values_to_root),
                )
            else:
                st.warning(f"{name}: Unsupported format ({ext})")

            all_chunks.extend(cur)
            prog.progress(idx / total)

        status.empty()
        prog.empty()

        if not all_chunks:
            st.warning("No chunks were produced.")
        else:
            # Text normalization
            if to_lower or to_upper:
                for ch in all_chunks:
                    if output_text_column in ch:
                        if to_upper:
                            ch[output_text_column] = ch[output_text_column].upper()
                        elif to_lower:
                            ch[output_text_column] = ch[output_text_column].lower()
                st.info(f"✅ Text normalization applied: {'UPPERCASE' if to_upper else 'lowercase'}")

            # Embeddings
            try:
                texts = [c[output_text_column] for c in all_chunks]
                emb_list: List[np.ndarray] = []
                bs = 32
                ep = st.progress(0)
                for i in range(0, len(texts), bs):
                    emb_list.append(
                        model.encode(
                            texts[i : i + bs],
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                    )
                    ep.progress(min((i + bs) / max(1, len(texts)), 1.0))
                ep.empty()
                embeddings = np.vstack(emb_list)
            except Exception as e:
                log_exception(e, "Embedding generation error: ")
                st.error(f"Embedding generation failed: {e}")
                for k in ("processed_chunks", "processed_embeddings", "results_ready"):
                    st.session_state.pop(k, None)
            else:
                emid = mcfg.get("model_id", "")
                for c in all_chunks:
                    c["embedding_model_id"] = emid
                    c["embedding_model_name"] = model_name
                    c["output_text_column"] = output_text_column

                st.session_state["processed_chunks"] = all_chunks
                st.session_state["processed_embeddings"] = embeddings
                st.session_state["last_model_used"] = model_name
                st.session_state["embedding_model_id"] = emid
                st.session_state["output_text_column"] = output_text_column
                st.session_state["results_ready"] = True
                st.rerun()

    # ===== PERSISTENT RESULTS =====
    if (
        st.session_state.get("results_ready")
        and "processed_chunks" in st.session_state
        and "processed_embeddings" in st.session_state
    ):
        all_chunks = st.session_state["processed_chunks"]
        embeddings = st.session_state["processed_embeddings"]
        out_col = st.session_state.get("output_text_column", "text")

        st.success("✅ Embedding generation complete!")
        st.markdown("---")
        st.subheader("🔎 Preview & Statistics")

        # Stats row 1
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Chunks", len(all_chunks))
        with s2:
            tab_n = sum(1 for c in all_chunks if c.get("is_tabular"))
            st.metric("Table Chunks", tab_n)
        with s3:
            st.metric("Text Chunks", len(all_chunks) - tab_n)
        with s4:
            toks = [c.get("token_count", 0) for c in all_chunks if c.get("token_count")]
            st.metric("Avg Tokens", f"{np.mean(toks):.1f}" if toks else "0")

        # Stats row 2
        s5, s6, s7, s8 = st.columns(4)
        with s5:
            st.metric("Sentence-Aware", sum(1 for c in all_chunks if c.get("sentence_aware")))
        with s6:
            st.metric("Single-Row Chunks", sum(1 for c in all_chunks if c.get("single_row")))
        with s7:
            st.metric("Split Rows", sum(1 for c in all_chunks if c.get("row_split")))
        with s8:
            st.metric("Files Processed", len({c["file"] for c in all_chunks}))

        # Preview table
        rows: List[Dict[str, Any]] = []
        for c in all_chunks[:200]:
            txt = c.get(out_col, "")
            r: Dict[str, Any] = {
                "file": c["file"],
                "source": c["source"],
                "row": str(c.get("row", "-")),
                "part": c["part"],
                "tokens": c.get("token_count", 0),
                "type": "Table" if c.get("is_tabular") else "Text",
                out_col: (txt[:150] + "…") if len(txt) > 150 else txt,
            }
            if c.get("sentence_aware"):
                r["feature"] = "Sentence-Aware"
            elif c.get("single_row"):
                r["feature"] = "Single Row"
            elif c.get("row_split"):
                r["feature"] = "Split"
            else:
                r["feature"] = "-"
            rows.append(r)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Downloads
        st.subheader("💾 Download")
        d1, d2, d3, d4 = st.columns(4)

        with d1:
            buf = io.BytesIO()
            np.save(buf, embeddings)
            buf.seek(0)
            st.download_button(
                "⬇️ Embeddings (.npy)", buf,
                file_name="embeddings.npy", mime="application/octet-stream", key="dl_emb",
            )

        with d2:
            jb = io.BytesIO()
            jb.write(json.dumps(all_chunks, ensure_ascii=False, indent=2).encode("utf-8"))
            jb.seek(0)
            st.download_button(
                "⬇️ Chunks (.json)", jb,
                file_name="chunks.json", mime="application/json", key="dl_json",
            )

        with d3:
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("chunks.json", json.dumps(all_chunks, ensure_ascii=False, indent=2).encode("utf-8"))
                nb = io.BytesIO()
                np.save(nb, embeddings)
                nb.seek(0)
                zf.writestr("embeddings.npy", nb.read())
            zb.seek(0)
            st.download_button(
                "📦 JSON + NPY (.zip)", zb,
                file_name="chunks_and_embeddings.zip", mime="application/zip", key="dl_zip",
            )

        with d4:
            if st.button("🧹 Clear Results"):
                for k in (
                    "processed_chunks", "processed_embeddings", "results_ready",
                    "last_model_used", "embedding_model_id", "output_text_column",
                ):
                    st.session_state.pop(k, None)
                st.rerun()

        st.info(
            """
💡 **Tips:**
- **NPY** contains the embedding vectors
- **JSON** contains text chunks and metadata
- **ZIP** bundles both files for convenience
- Results persist across page refreshes until you click *Clear Results*
- Text normalization (lowercase / uppercase) is applied to chunk text
            """
        )


if __name__ == "__main__":
    main()
"""
Configuration for Document Chunking and Embedding Application
"""

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2 (English, Fast)": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 256,
        "language": "English",
        "description": "Fast and lightweight English embedding model (384 dim)",
    },
    "all-MiniLM-L12-v2 (English, Balanced)": {
        "model_id": "sentence-transformers/all-MiniLM-L12-v2",
        "max_length": 256,
        "language": "English",
        "description": "Balanced English embedding model (384 dim)",
    },
    "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "max_length": 128,
        "language": "Multilingual (50+ languages incl. Turkish)",
        "description": "Multilingual embedding model with Turkish support (384 dim)",
    },
    "paraphrase-multilingual-mpnet-base-v2 (Multilingual, High Quality)": {
        "model_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "max_length": 128,
        "language": "Multilingual (50+ languages incl. Turkish)",
        "description": "High-quality multilingual embedding model with Turkish support (768 dim)",
    },
    "all-mpnet-base-v2 (English, High Quality)": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "max_length": 384,
        "language": "English",
        "description": "High-quality English embedding model (768 dim)",
    },
}

APP_CONFIG = {
    "title": "📄 Document Chunking & Embedding",
    "supported_formats": ["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    "max_files": 20,
    "default_values": {
        "max_tokens": 128,
        "overlap": 16,
        "values_only_threshold": 2,
        "min_cols_for_table": 2,
        "min_rows_for_table": 2,
        "include_column_names": True,
        "attach_row_data": True,
        "flatten_row_values_to_root": True,
    },
}


def get_model_config(model_name: str) -> dict:
    """Return config dict for the given model display name."""
    if model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]
    first_key = next(iter(EMBEDDING_MODELS))
    return EMBEDDING_MODELS[first_key]
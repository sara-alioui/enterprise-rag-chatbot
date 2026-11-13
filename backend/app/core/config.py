"""Configuration centrale : tous les réglages du projet au même endroit.

Chaque valeur peut être remplacée par une variable d'environnement RAG_<NOM>,
par exemple RAG_LLM_MODEL=qwen2.5:3b.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # racine du projet


@dataclass(frozen=True)
class Settings:
    # --- chemins ---
    docs_dir: Path = ROOT / "docs"
    chroma_dir: Path = ROOT / "chroma_db"
    state_file: Path = ROOT / "ingest_state.json"
    collection_name: str = "docs"

    # --- embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"

    # --- découpage ---
    chunk_max_words: int = 180
    chunk_min_words: int = 8
    chunk_overlap_lines: int = 2

    # --- OCR (texte des images et des PDF scannés) ---
    ocr_enabled: bool = True
    ocr_languages: str = "fra+eng"
    ocr_min_page_words: int = 15   # en dessous, la page est considérée comme scannée
    ocr_min_image_side: int = 200  # ignore les petites images (logos, icônes)

    # --- recherche ---
    top_k: int = 3
    neighbors: int = 1
    max_passages: int = 9
    keyword_max_terms: int = 3
    keyword_per_term: int = 2

    # --- modèle de langage (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout: int = 900

    # --- API ---
    cors_origins: tuple = ("http://localhost:3000", "http://localhost:5173",
                           "http://localhost:8501")
    max_upload_mb: int = 50


def _convert(raw: str, default):
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "oui"}
    if isinstance(default, Path):
        return Path(raw)
    if isinstance(default, tuple):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return type(default)(raw)


def load_settings() -> Settings:
    """Réglages par défaut, surchargés par les variables d'environnement RAG_*."""
    base = Settings()
    overrides = {}
    for f in fields(Settings):
        raw = os.getenv(f"RAG_{f.name.upper()}")
        if raw is not None:
            overrides[f.name] = _convert(raw, getattr(base, f.name))
    return replace(base, **overrides)


@lru_cache
def get_settings() -> Settings:
    return load_settings()

"""Point d'entrée unique pour lire un document, quel que soit son format."""
from __future__ import annotations

from pathlib import Path

from backend.app.core.config import Settings
from backend.app.database.models import Page
from backend.app.ingestion.docx_loader import load_docx
from backend.app.ingestion.ocr import ocr_image_bytes
from backend.app.ingestion.pdf_loader import load_pdf

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx"} | IMAGE_EXTENSIONS


def is_supported(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_document(path: Path, settings: Settings) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, settings)
    if ext == ".docx":
        return load_docx(path)
    if ext in IMAGE_EXTENSIONS:
        return [Page(0, ocr_image_bytes(path.read_bytes(), settings.ocr_languages, ext))]
    raise ValueError(f"Format non pris en charge : {ext}")

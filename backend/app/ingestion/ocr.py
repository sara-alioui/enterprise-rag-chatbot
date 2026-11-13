"""OCR : lire le texte contenu dans une image avec Tesseract."""
from __future__ import annotations

import shutil
import subprocess
import tempfile


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_image_bytes(data: bytes, languages: str = "fra+eng", suffix: str = ".png") -> str:
    """Texte reconnu dans une image ('' si Tesseract est absent ou échoue)."""
    if not tesseract_available():
        return ""
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        result = subprocess.run(["tesseract", f.name, "stdout", "-l", languages],
                                capture_output=True, text=True)
    return result.stdout.strip()

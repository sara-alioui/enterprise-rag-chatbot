"""Lecture des PDF : texte, pages scannées (OCR) et texte des images intégrées."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from backend.app.database.models import Page
from backend.app.ingestion.ocr import ocr_image_bytes

log = get_logger(__name__)


def _ocr_embedded_images(doc, page, settings: Settings) -> list[str]:
    found = []
    for img in page.get_images(full=True):
        try:
            pix = pymupdf.Pixmap(doc, img[0])
            if min(pix.width, pix.height) < settings.ocr_min_image_side:
                continue
            if pix.colorspace is None or pix.colorspace.n > 3 or pix.alpha:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            text = ocr_image_bytes(pix.tobytes("png"), settings.ocr_languages)
            if len(text.split()) >= 5:
                found.append(text)
        except Exception as exc:  # image illisible : on l'ignore
            log.debug("image ignorée : %s", exc)
    return found


def load_pdf(path: Path, settings: Settings) -> list[Page]:
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            if settings.ocr_enabled:
                if len(text.split()) < settings.ocr_min_page_words:
                    # page scannée : OCR de la page entière
                    ocr = ocr_image_bytes(page.get_pixmap(dpi=200).tobytes("png"),
                                          settings.ocr_languages)
                    if len(ocr.split()) > len(text.split()):
                        text = ocr
                else:
                    extra = _ocr_embedded_images(doc, page, settings)
                    if extra:
                        text += "\n[Texte dans une image]\n" + "\n".join(extra)
            pages.append(Page(number, text))
    return pages

"""Découpage du texte en passages (par lignes entières, avec chevauchement)."""
from __future__ import annotations

import re


def chunk_text(text: str, max_words: int = 180, min_words: int = 8,
               overlap_lines: int = 2) -> list[str]:
    """Regroupe des lignes entières jusqu'à max_words mots.

    Le code et les énoncés ne sont jamais coupés au milieu d'une ligne.
    Les `overlap_lines` dernières lignes sont reprises au début du passage suivant.
    """
    units = [re.sub(r"[ \t]+", " ", u).strip() for u in text.split("\n")]
    units = [u for u in units if u]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = len(unit.split())
        if current and count + words > max_words:
            chunks.append("\n".join(current))
            current = current[-overlap_lines:] if overlap_lines > 0 else []
            count = sum(len(x.split()) for x in current)
        current.append(unit)
        count += words
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if len(c.split()) >= min_words]

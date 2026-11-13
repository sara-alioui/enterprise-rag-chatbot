"""Sélection des preuves : ne garder que les passages qui soutiennent réellement la réponse."""
from __future__ import annotations

import re
import unicodedata

from backend.app.database.models import RetrievedChunk


def _norm_words(text: str) -> set[str]:
    decomposed = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"\w+", plain) if len(w) >= 4}


def support_score(chunk_text: str, citation: str, answer: str) -> float:
    """Part des mots (>= 4 lettres) de la citation et de la réponse présents dans le passage.
    La citation compte double : c'est elle qui désigne l'endroit exact."""
    words = _norm_words(chunk_text)
    cit, ans = _norm_words(citation), _norm_words(answer)
    if not cit and not ans:
        return 0.0
    cit_part = len(cit & words) / len(cit) if cit else 0.0
    ans_part = len(ans & words) / len(ans) if ans else 0.0
    return (2 * cit_part + ans_part) / 3


def select_evidence(passages: list[RetrievedChunk], citation: str, answer: str,
                    max_sources: int = 3, min_score: float = 0.5) -> list[RetrievedChunk]:
    """Passages les plus fidèles à la citation/réponse (au plus `max_sources`),
    remis dans l'ordre de lecture. Si aucun n'atteint le seuil, on garde le meilleur."""
    scored = sorted(((support_score(p.chunk.text, citation, answer), p) for p in passages),
                    key=lambda t: t[0], reverse=True)
    if not scored:
        return []
    kept = [p for s, p in scored[:max_sources] if s >= min_score] or [scored[0][1]]
    return sorted(kept, key=lambda p: (p.chunk.source, p.chunk.index))

"""Fusion des résultats de recherche : déduplication, limite et ordre de lecture."""
from __future__ import annotations

from backend.app.database.models import RetrievedChunk


def merge_results(keyword: list[RetrievedChunk], vector: list[RetrievedChunk],
                  neighbors: list[RetrievedChunk], max_passages: int) -> list[RetrievedChunk]:
    """Priorité : mots-clés, puis similarité, puis voisins.

    Les doublons sont supprimés, la liste est limitée à `max_passages`, puis triée
    dans l'ordre du document (le modèle lit le texte dans l'ordre naturel).
    """
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for group in (keyword, vector, neighbors):
        for item in group:
            if item.chunk.id not in seen:
                seen.add(item.chunk.id)
                merged.append(item)
    merged = merged[:max_passages]
    merged.sort(key=lambda r: (r.chunk.source, r.chunk.index))
    return merged

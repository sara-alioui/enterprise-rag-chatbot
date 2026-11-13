"""Recherche hybride : par sens (vecteurs) + par mots précis + passages voisins."""
from __future__ import annotations

import re

from backend.app.core.config import Settings
from backend.app.database.models import RetrievedChunk, make_chunk_id
from backend.app.embeddings.embedding_model import EmbeddingModel
from backend.app.retrieval.reranker import merge_results
from backend.app.vectorstore.vector_store import VectorStore


def extract_key_terms(question: str) -> list[str]:
    """Mots précis à chercher tels quels : identifiants de code et noms propres.

    Exemples : custom_cooling, getX2, 0.99, Capgemini, Jedi.
    Les mots qui commencent une phrase (Quelle, Comment...) sont ignorés.
    """
    terms: list[str] = []
    for m in re.finditer(r"\w+", question):
        word = m.group()
        before = question[:m.start()].rstrip()
        sentence_start = before == "" or before[-1] in ".?!:"
        is_code = ("_" in word or any(ch.isdigit() for ch in word)
                   or re.search(r"[a-z][A-Z]", word))
        is_name = (not sentence_start) and word[0].isupper() and len(word) >= 4
        if (is_code or is_name) and word not in terms:
            terms.append(word)
    return terms


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingModel, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def _keyword_hits(self, question: str, source: str | None) -> list[RetrievedChunk]:
        hits: list[RetrievedChunk] = []
        seen: set[str] = set()
        for term in extract_key_terms(question)[: self.settings.keyword_max_terms]:
            candidates = self.store.keyword_search(term, source)
            candidates.sort(key=lambda c: -c.text.count(term))
            for chunk in candidates[: self.settings.keyword_per_term]:
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    hits.append(RetrievedChunk(chunk, 1.0, "keyword"))
        return hits

    def _neighbors(self, primary: list[RetrievedChunk], width: int) -> list[RetrievedChunk]:
        have = {r.chunk.id for r in primary}
        wanted: list[str] = []
        for r in primary:
            for d in range(1, width + 1):
                for index in (r.chunk.index - d, r.chunk.index + d):
                    cid = make_chunk_id(r.chunk.source, index)
                    if index >= 0 and cid not in have and cid not in wanted:
                        wanted.append(cid)
        return [RetrievedChunk(c, 0.0, "neighbor") for c in self.store.get_by_ids(wanted)]

    def retrieve(self, question: str, k: int | None = None, neighbors: int | None = None,
                 source: str | None = None) -> list[RetrievedChunk]:
        if self.store.count() == 0:   # base vide : inutile de charger le modèle
            return []
        k = self.settings.top_k if k is None else k
        width = self.settings.neighbors if neighbors is None else neighbors
        keyword = self._keyword_hits(question, source)
        vector = self.store.query(self.embedder.embed_query(question), k, source)
        primary = merge_results(keyword, vector, [], max_passages=10_000)
        extra = self._neighbors(primary, width) if width else []
        return merge_results(keyword, vector, extra, self.settings.max_passages)

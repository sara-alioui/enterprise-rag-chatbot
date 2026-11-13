"""Interface commune à toutes les bases vectorielles (ChromaDB, FAISS, ...)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.database.models import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def delete_source(self, source: str) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], k: int,
              source: str | None = None) -> list[RetrievedChunk]:
        """Les k passages les plus proches (score = similarité cosinus)."""

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Passages demandés, dans l'ordre des ids (ids absents ignorés)."""

    @abstractmethod
    def keyword_search(self, term: str, source: str | None = None,
                       limit: int = 200) -> list[Chunk]:
        """Passages contenant exactement le mot (sensible à la casse)."""

    @abstractmethod
    def list_documents(self) -> dict[str, int]:
        """Nom de chaque document -> nombre de passages."""

    @abstractmethod
    def count(self) -> int: ...

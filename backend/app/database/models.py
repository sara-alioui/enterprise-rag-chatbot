"""Objets métier partagés par tous les modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def document_title(source: str) -> str:
    """Nom lisible d'un document (utilisé pour enrichir les embeddings)."""
    return Path(source).stem.replace("_", " ")


@dataclass(frozen=True)
class Page:
    number: int  # 0 si le format n'a pas de pages (Word, image)
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str  # nom du fichier
    index: int   # numéro du passage dans le document
    page: int
    text: str

    @property
    def id(self) -> str:
        return make_chunk_id(self.source, self.index)


def make_chunk_id(source: str, index: int) -> str:
    return f"{source}::{index}"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    origin: str  # "vector" | "keyword" | "neighbor"


@dataclass
class Answer:
    answered: bool                       # False = « Je ne trouve pas la réponse »
    text: str
    citation: str = ""
    sources: list[RetrievedChunk] = field(default_factory=list)
    note: str = ""                       # explication quand la réponse est écartée

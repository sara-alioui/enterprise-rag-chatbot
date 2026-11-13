"""Formats d'échange de l'API (entrées et sorties)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(max_length=2000)
    document: str | None = Field(default=None, description="Limiter la recherche à ce document")
    k: int | None = Field(default=None, ge=1, le=10, description="Passages les plus proches")
    neighbors: int | None = Field(default=None, ge=0, le=3, description="Voisins de chaque passage")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La question est vide.")
        return value


class SourceOut(BaseModel):
    document: str
    passage: int
    page: int
    origin: str          # "vector" | "keyword" | "neighbor"
    score: float
    text: str


class ChatResponse(BaseModel):
    answered: bool
    answer: str
    citation: str = ""
    note: str = ""
    sources: list[SourceOut] = []


class DocumentOut(BaseModel):
    name: str
    passages: int


class UploadResult(BaseModel):
    name: str
    status: str          # "indexé" | "vide" | "refusé" | "erreur"
    passages: int = 0
    detail: str = ""


class HealthOut(BaseModel):
    status: str
    documents: int
    passages: int
    llm_model: str
    llm_reachable: bool

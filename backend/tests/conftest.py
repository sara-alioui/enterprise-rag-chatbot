"""Outils de test : aucun téléchargement de modèle, tout tourne en quelques secondes."""
import hashlib
import re
from dataclasses import replace

import pytest

from backend.app.core.config import Settings
from backend.app.database.database import IngestionState
from backend.app.ingestion.service import IngestionService
from backend.app.vectorstore.chroma_store import ChromaStore

DIM = 256


class FakeEmbedder:
    """Faux modèle d'embeddings : sac de mots haché. Deux textes qui partagent des mots
    ont des vecteurs proches, ce qui suffit pour tester la logique de recherche."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts, title=""):
        return [self._vec(f"{title} {t}") for t in texts]

    def embed_query(self, query):
        return self._vec(query)


class FakeLLM:
    """Faux modèle de langage : renvoie la sortie préparée par le test."""

    def __init__(self, output: str = ""):
        self.output = output
        self.calls = 0

    def chat(self, system, user, json_schema=None):
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output

    def is_available(self):
        return not isinstance(self.output, Exception)


@pytest.fixture
def settings(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return replace(Settings(), docs_dir=docs, chroma_dir=tmp_path / "chroma",
                   state_file=tmp_path / "state.json", ocr_enabled=False,
                   chunk_max_words=30, chunk_min_words=3)


@pytest.fixture
def store(settings):
    return ChromaStore(settings.chroma_dir, "test")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def ingestion(settings, store, embedder):
    return IngestionService(store, embedder, IngestionState(settings.state_file), settings)

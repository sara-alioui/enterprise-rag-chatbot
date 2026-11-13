"""Modèle d'embeddings (vecteurs de sens) : multilingual-e5 par défaut."""
from __future__ import annotations

from backend.app.database.models import document_title  # noqa: F401  (réexporté)


class EmbeddingModel:
    """Enveloppe SentenceTransformer, chargée à la première utilisation.

    Les modèles e5 demandent un préfixe : « passage: » pour les textes indexés,
    « query: » pour les questions. Le titre du document est ajouté devant chaque
    passage afin que la question puisse aussi correspondre au nom du fichier.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        is_e5 = "e5" in model_name.lower()
        self.passage_prefix = "passage: " if is_e5 else ""
        self.query_prefix = "query: " if is_e5 else ""

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str], title: str = "") -> list[list[float]]:
        head = f"{self.passage_prefix}{title}\n" if title else self.passage_prefix
        vectors = self._load().encode([head + t for t in texts],
                                      normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self._load().encode(self.query_prefix + query, normalize_embeddings=True)
        return vector.tolist()

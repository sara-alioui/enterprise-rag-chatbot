"""Assemblage des composants : un seul endroit où tout est branché."""
from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.config import Settings, get_settings
from backend.app.database.database import IngestionState
from backend.app.embeddings.embedding_model import EmbeddingModel
from backend.app.ingestion.service import IngestionService
from backend.app.llm.llm_client import OllamaClient
from backend.app.rag.rag_pipeline import RAGPipeline
from backend.app.retrieval.retriever import HybridRetriever
from backend.app.vectorstore.chroma_store import ChromaStore


@dataclass
class Services:
    settings: Settings
    store: ChromaStore
    embedder: EmbeddingModel
    ingestion: IngestionService
    rag: RAGPipeline


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    store = ChromaStore(settings.chroma_dir, settings.collection_name)
    embedder = EmbeddingModel(settings.embedding_model)
    state = IngestionState(settings.state_file)
    ingestion = IngestionService(store, embedder, state, settings)
    rag = RAGPipeline(HybridRetriever(store, embedder, settings), OllamaClient(settings))
    return Services(settings, store, embedder, ingestion, rag)

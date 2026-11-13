"""Ingestion : lire, découper, vectoriser et stocker les documents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.chunking.text_chunker import chunk_text
from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from backend.app.database.database import IngestionState
from backend.app.database.models import Chunk, document_title
from backend.app.embeddings.embedding_model import EmbeddingModel
from backend.app.ingestion.document_loader import is_supported, load_document
from backend.app.ingestion.text_cleaner import clean_text
from backend.app.vectorstore.vector_store import VectorStore

log = get_logger(__name__)


@dataclass
class IngestResult:
    name: str
    status: str   # "indexé" | "inchangé" | "supprimé" | "vide"
    passages: int = 0


class IngestionService:
    def __init__(self, store: VectorStore, embedder: EmbeddingModel,
                 state: IngestionState, settings: Settings):
        self.store, self.embedder, self.state, self.settings = store, embedder, state, settings

    def ingest_file(self, path: Path) -> int:
        """Indexe (ou réindexe) un fichier. Retourne le nombre de passages créés."""
        path = Path(path)
        name = path.name
        self.store.delete_source(name)  # remplace l'ancienne version
        s = self.settings
        chunks: list[Chunk] = []
        for page in load_document(path, s):
            for text in chunk_text(clean_text(page.text), s.chunk_max_words,
                                   s.chunk_min_words, s.chunk_overlap_lines):
                chunks.append(Chunk(name, len(chunks), page.number, text))
        if not chunks:
            self.state.remove(name)
            log.warning("%s : aucun texte trouvé", name)
            return 0
        embeddings = self.embedder.embed_passages([c.text for c in chunks],
                                                  title=document_title(name))
        self.store.add_chunks(chunks, embeddings)
        self.state.set(name, IngestionState.file_hash(path))
        return len(chunks)

    def remove_document(self, name: str) -> None:
        self.store.delete_source(name)
        (self.settings.docs_dir / name).unlink(missing_ok=True)
        self.state.remove(name)

    def sync_folder(self) -> list[IngestResult]:
        """Aligne la base sur le dossier docs/ : nouveaux, modifiés, supprimés."""
        self.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        files = {f.name: f for f in sorted(self.settings.docs_dir.iterdir())
                 if f.is_file() and is_supported(f)}
        report: list[IngestResult] = []
        for name in self.state.names():
            if name not in files:
                self.store.delete_source(name)
                self.state.remove(name)
                report.append(IngestResult(name, "supprimé"))
        for name, path in files.items():
            if self.state.get(name) == IngestionState.file_hash(path):
                report.append(IngestResult(name, "inchangé"))
                continue
            n = self.ingest_file(path)
            report.append(IngestResult(name, "indexé" if n else "vide", n))
        return report

    def list_documents(self) -> dict[str, int]:
        return self.store.list_documents()

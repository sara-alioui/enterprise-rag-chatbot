"""Base vectorielle persistante ChromaDB."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from backend.app.database.models import Chunk, RetrievedChunk
from backend.app.vectorstore.vector_store import VectorStore


def _chunk(doc: str, meta: dict) -> Chunk:
    return Chunk(source=meta["source"], index=meta["chunk_id"],
                 page=meta.get("page", 0), text=doc)


class ChromaStore(VectorStore):
    def __init__(self, path: Path, collection_name: str = "docs"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks, embeddings):
        for s in range(0, len(chunks), 500):
            part = slice(s, s + 500)
            self._col.add(
                ids=[c.id for c in chunks[part]],
                embeddings=embeddings[part],
                documents=[c.text for c in chunks[part]],
                metadatas=[{"source": c.source, "chunk_id": c.index, "page": c.page}
                           for c in chunks[part]],
            )

    def delete_source(self, source):
        self._col.delete(where={"source": source})

    def query(self, embedding, k, source=None):
        total = self._col.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding], n_results=min(k, total),
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"])
        return [RetrievedChunk(_chunk(d, m), 1.0 - dist, "vector")
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                      res["distances"][0])]

    def get_by_ids(self, ids):
        if not ids:
            return []
        res = self._col.get(ids=ids, include=["documents", "metadatas"])
        by_id = {i: _chunk(d, m) for i, d, m in
                 zip(res["ids"], res["documents"], res["metadatas"])}
        return [by_id[i] for i in ids if i in by_id]

    def keyword_search(self, term, source=None, limit=200):
        res = self._col.get(where_document={"$contains": term},
                            where={"source": source} if source else None,
                            include=["documents", "metadatas"], limit=limit)
        return [_chunk(d, m) for d, m in zip(res["documents"], res["metadatas"])]

    def list_documents(self):
        metas = self._col.get(include=["metadatas"])["metadatas"]
        return dict(Counter(m["source"] for m in metas))

    def count(self):
        return self._col.count()

"""Base vectorielle FAISS : à implémenter (alternative à ChromaDB).

Pour l'ajouter, créer une classe FaissStore(VectorStore) qui implémente les mêmes
méthodes que ChromaStore. Le reste du projet n'a pas besoin de changer.
"""
from backend.app.vectorstore.vector_store import VectorStore


class FaissStore(VectorStore):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FaissStore n'est pas encore implémenté.")

    add_chunks = delete_source = query = get_by_ids = keyword_search = None  # type: ignore
    list_documents = count = None  # type: ignore

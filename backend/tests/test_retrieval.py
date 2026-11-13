from backend.app.database.models import Chunk, RetrievedChunk
from backend.app.retrieval.reranker import merge_results
from backend.app.retrieval.retriever import HybridRetriever, extract_key_terms


# ---------- mots précis ----------
def test_key_terms_finds_identifiers_names_and_numbers():
    assert extract_key_terms("Que fait custom_cooling ?") == ["custom_cooling"]
    assert extract_key_terms("Qu'est-ce qu'un Jedi dans l'exercice ?") == ["Jedi"]
    assert "getX2" in extract_key_terms("Que retourne getX2 ?")


def test_key_terms_ignores_sentence_openers():
    assert extract_key_terms("Quelle est la capitale ? Comment faire ?") == []
    assert extract_key_terms("À qui la lettre est-elle adressée ?") == []


# ---------- fusion ----------
def _rc(source, index, origin="vector"):
    return RetrievedChunk(Chunk(source, index, 0, f"t{index}"), 0.5, origin)


def test_merge_dedupes_caps_and_orders_by_document():
    kw = [_rc("a", 5, "keyword")]
    vec = [_rc("a", 5), _rc("b", 1), _rc("a", 2)]
    nb = [_rc("a", 3, "neighbor"), _rc("a", 4, "neighbor")]
    merged = merge_results(kw, vec, nb, max_passages=4)
    # priorité : mot-clé (a5), puis vecteurs (b1, a2), puis voisins (a3) ; a4 est coupé
    assert [(r.chunk.source, r.chunk.index) for r in merged] == \
        [("a", 2), ("a", 3), ("a", 5), ("b", 1)]


# ---------- recherche hybride ----------
def _index(store, embedder, source, texts):
    chunks = [Chunk(source, i, 0, t) for i, t in enumerate(texts)]
    store.add_chunks(chunks, embedder.embed_passages([c.text for c in chunks]))


def test_vector_search_returns_closest_passage(settings, store, embedder):
    _index(store, embedder, "doc.pdf", ["le chat dort sur le canapé", "la voiture roule vite sur la route",
                                         "la recette de cuisine demande du sucre"])
    r = HybridRetriever(store, embedder, settings).retrieve("voiture sur la route", k=1, neighbors=0)
    assert r[0].chunk.text.startswith("la voiture")


def test_keyword_search_finds_exact_identifier_that_vectors_miss(settings, store, embedder):
    texts = ["introduction générale du projet et du contexte"] * 1 + \
            ["function custom_cooling initial_temp k max_iter"] + \
            ["conclusion des travaux et perspectives futures"]
    _index(store, embedder, "recuit.pdf", texts)
    found = HybridRetriever(store, embedder, settings).retrieve(
        "Que fait custom_cooling ?", k=1, neighbors=0)
    assert any(r.origin == "keyword" and "custom_cooling" in r.chunk.text for r in found)


def test_neighbors_are_added_within_same_document_only(settings, store, embedder):
    _index(store, embedder, "a.pdf", ["alpha un", "bravo deux unique", "charlie trois"])
    _index(store, embedder, "b.pdf", ["delta quatre", "echo cinq"])
    r = HybridRetriever(store, embedder, settings).retrieve("bravo unique", k=1, neighbors=1)
    got = {(x.chunk.source, x.chunk.index) for x in r}
    assert ("a.pdf", 1) in got and ("a.pdf", 0) in got and ("a.pdf", 2) in got
    assert not any(src == "b.pdf" for src, _ in got)
    assert {x.origin for x in r if x.chunk.index != 1} == {"neighbor"}


def test_source_filter(settings, store, embedder):
    _index(store, embedder, "a.pdf", ["le chat dort"])
    _index(store, embedder, "b.pdf", ["le chat mange"])
    r = HybridRetriever(store, embedder, settings).retrieve("chat", k=5, neighbors=0, source="b.pdf")
    assert {x.chunk.source for x in r} == {"b.pdf"}


def test_empty_store_returns_nothing(settings, store, embedder):
    assert HybridRetriever(store, embedder, settings).retrieve("question") == []


def test_empty_store_does_not_load_the_embedding_model(settings, store):
    class Exploding:
        def embed_query(self, q):
            raise AssertionError("le modèle ne doit pas être chargé")
    assert HybridRetriever(store, Exploding(), settings).retrieve("question") == []

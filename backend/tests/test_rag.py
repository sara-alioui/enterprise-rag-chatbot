import json

from backend.app.database.models import Chunk
from backend.app.llm.prompt_templates import NO_ANSWER
from backend.app.rag.answer_verifier import parse_llm_json, verify_answer
from backend.app.rag.rag_pipeline import RAGPipeline
from backend.app.retrieval.retriever import HybridRetriever
from backend.tests.conftest import FakeLLM

CONTEXT = ("[Extrait 1 - lettre.pdf]\nÀ l'attention de l'équipe Recrutement\n"
           "Capgemini Engineering – Tunisie\nstagetunisie.tn@capgemini.com\n"
           "Fès, Maroc")


def _raw(citation, reponse):
    return json.dumps({"citation": citation, "reponse": reponse})


# ---------- vérification des réponses ----------
def test_valid_answer_is_accepted():
    v = verify_answer(_raw("Capgemini Engineering – Tunisie",
                           "La lettre est adressée à Capgemini Engineering – Tunisie."), CONTEXT)
    assert v.ok and "Capgemini" in v.answer


def test_invented_proper_noun_is_rejected_even_with_real_quote():
    # cas réel : citation authentique mais sans rapport, « Rabat » inventé
    v = verify_answer(_raw("Capgemini Engineering – Tunisie",
                           "La capitale du Maroc est Rabat."), CONTEXT)
    assert not v.ok and v.reason == "unsupported_terms" and v.unsupported == ["Rabat"]


def test_invented_number_is_rejected():
    v = verify_answer(_raw("Capgemini Engineering", "Il possède une vitesse de 5km/h."), CONTEXT)
    assert not v.ok and "5km" in v.unsupported


def test_made_up_citation_is_rejected():
    v = verify_answer(_raw("La capitale du royaume est une grande ville", "C'est une ville."), CONTEXT)
    assert not v.ok and v.reason == "citation_not_in_context"


def test_empty_citation_or_refusal_is_rejected():
    assert verify_answer(_raw("", "Quelque chose."), CONTEXT).reason == "no_citation"
    assert verify_answer(_raw("Capgemini Engineering", NO_ANSWER), CONTEXT).reason == "no_citation"


def test_accents_and_case_do_not_matter():
    v = verify_answer(_raw("equipe recrutement", "Elle va à l'Équipe Recrutement."), CONTEXT)
    assert v.ok


def test_malformed_json_is_rejected_and_embedded_json_is_recovered():
    assert verify_answer("pas du json", CONTEXT).reason == "invalid_json"
    assert parse_llm_json('Voici : {"citation": "a", "reponse": "b"} fin') == ("a", "b")


# ---------- pipeline complet (faux modèle de langage) ----------
def _pipeline(settings, store, embedder, llm_output):
    chunks = [Chunk("lettre.pdf", 0, 1, "À l'attention de l'équipe Recrutement Capgemini Engineering Tunisie")]
    store.add_chunks(chunks, embedder.embed_passages([c.text for c in chunks]))
    llm = FakeLLM(llm_output)
    return RAGPipeline(HybridRetriever(store, embedder, settings), llm), llm


def test_pipeline_answers_with_sources(settings, store, embedder):
    rag, _ = _pipeline(settings, store, embedder,
                       _raw("équipe Recrutement Capgemini Engineering", "Elle est adressée à Capgemini Engineering."))
    answer = rag.ask("À qui est adressée la lettre ?")
    assert answer.answered and "Capgemini" in answer.text
    assert answer.citation and answer.sources


def test_pipeline_rejects_invention_and_hides_sources(settings, store, embedder):
    rag, _ = _pipeline(settings, store, embedder,
                       _raw("équipe Recrutement Capgemini Engineering", "La capitale est Rabat."))
    answer = rag.ask("Quelle est la capitale ?")
    assert not answer.answered and answer.text == NO_ANSWER
    assert "Rabat" in answer.note and answer.sources == []


def test_pipeline_without_documents_does_not_call_llm(settings, store, embedder):
    llm = FakeLLM("{}")
    rag = RAGPipeline(HybridRetriever(store, embedder, settings), llm)
    answer = rag.ask("Une question ?")
    assert not answer.answered and llm.calls == 0


# ---------- sélection des preuves ----------
def test_select_evidence_keeps_only_supporting_passages():
    from backend.app.database.models import RetrievedChunk
    from backend.app.rag.evidence import select_evidence
    mk = lambda i, t: RetrievedChunk(Chunk("d.pdf", i, 1, t), 0.8, "vector")
    ps = [mk(0, "Madame Monsieur bonjour"),
          mk(1, "Le projet Jedi atteint une vitesse maximale de douze kilometres"),
          mk(2, "Facture numero 1234 du client"),
          mk(3, "Contact: sara@example.com")]
    out = select_evidence(ps, "vitesse maximale de douze kilometres", "La vitesse maximale est de douze kilometres.")
    assert [p.chunk.index for p in out] == [1]


def test_select_evidence_falls_back_to_best_passage():
    from backend.app.database.models import RetrievedChunk
    from backend.app.rag.evidence import select_evidence
    ps = [RetrievedChunk(Chunk("d.pdf", 0, 1, "texte sans rapport"), 0.5, "vector")]
    assert len(select_evidence(ps, "autre chose", "autre")) == 1

import io
import json

import pytest
from docx import Document
from fastapi.testclient import TestClient

from backend.app.container import Services
from backend.app.llm.llm_client import LLMUnavailable
from backend.app.main import create_app
from backend.app.rag.rag_pipeline import RAGPipeline
from backend.app.retrieval.retriever import HybridRetriever
from backend.tests.conftest import FakeLLM

LETTRE = ("À l'attention de l'équipe Recrutement de Capgemini Engineering Tunisie pour le stage de fin d'études.\n"
          "Rigoureuse, curieuse et autonome, la candidate apprend vite de nouveaux outils.")


def _docx_bytes(text):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_client(settings, store, embedder, ingestion, llm):
    rag = RAGPipeline(HybridRetriever(store, embedder, settings), llm)
    app = create_app(Services(settings, store, embedder, ingestion, rag))
    return TestClient(app)


@pytest.fixture
def llm():
    return FakeLLM(json.dumps({"citation": "équipe Recrutement de Capgemini Engineering Tunisie",
                               "reponse": "Elle est adressée à Capgemini Engineering Tunisie."}))


@pytest.fixture
def client(settings, store, embedder, ingestion, llm):
    return _make_client(settings, store, embedder, ingestion, llm)


def _upload(client, name="lettre.docx", content=None):
    content = _docx_bytes(LETTRE) if content is None else content
    return client.post("/documents", files=[("files", (name, content))])


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_reachable"] is True and body["documents"] == 0


def test_health_reports_unreachable_llm(settings, store, embedder, ingestion):
    c = _make_client(settings, store, embedder, ingestion, FakeLLM(LLMUnavailable("x")))
    assert c.get("/health").json()["status"] == "dégradé"


def test_upload_list_chat_delete_flow(client, settings):
    result = _upload(client).json()[0]
    assert result["status"] == "indexé" and result["passages"] >= 1
    assert (settings.docs_dir / "lettre.docx").exists()

    assert client.get("/documents").json() == [{"name": "lettre.docx", "passages": result["passages"]}]

    chat = client.post("/chat", json={"question": "À qui la lettre est-elle adressée ?"}).json()
    assert chat["answered"] and "Capgemini" in chat["answer"]
    assert chat["citation"] and chat["sources"][0]["document"] == "lettre.docx"

    assert client.delete("/documents/lettre.docx").json() == {"deleted": "lettre.docx"}
    assert client.get("/documents").json() == []
    assert not (settings.docs_dir / "lettre.docx").exists()


def test_chat_on_one_document(client):
    _upload(client)
    ok = client.post("/chat", json={"question": "Capgemini ?", "document": "lettre.docx"})
    assert ok.status_code == 200
    assert client.post("/chat", json={"question": "Capgemini ?", "document": "inconnu.pdf"}).status_code == 404


def test_chat_without_documents_says_so(client):
    body = client.post("/chat", json={"question": "Une question ?"}).json()
    assert body["answered"] is False and "Aucun document" in body["note"]


def test_chat_rejects_invented_answer(settings, store, embedder, ingestion):
    llm = FakeLLM(json.dumps({"citation": "équipe Recrutement de Capgemini Engineering Tunisie",
                              "reponse": "La capitale est Rabat."}))
    c = _make_client(settings, store, embedder, ingestion, llm)
    _upload(c)
    body = c.post("/chat", json={"question": "Quelle est la capitale ?"}).json()
    assert body["answered"] is False and "Rabat" in body["note"] and body["sources"] == []


def test_chat_returns_503_when_llm_is_down(settings, store, embedder, ingestion):
    c = _make_client(settings, store, embedder, ingestion, FakeLLM(LLMUnavailable("Ollama ne répond pas.")))
    _upload(c)
    response = c.post("/chat", json={"question": "Une question ?"})
    assert response.status_code == 503 and "Ollama" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}, {"question": "a", "k": 0}])
def test_chat_validates_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_upload_refuses_unsupported_format(client):
    result = _upload(client, "notes.txt", b"bonjour").json()[0]
    assert result["status"] == "refusé"


def test_upload_strips_path_from_filename(client, settings):
    _upload(client, "../../evil.docx")
    assert (settings.docs_dir / "evil.docx").exists()
    assert not (settings.docs_dir.parent.parent / "evil.docx").exists()


def test_upload_corrupt_file_is_reported_not_fatal(client, settings):
    result = _upload(client, "cassé.pdf", b"ceci n'est pas un pdf").json()[0]
    assert result["status"] == "erreur" and not (settings.docs_dir / "cassé.pdf").exists()


def test_delete_unknown_and_invalid_names(client):
    assert client.delete("/documents/absent.pdf").status_code == 404
    assert client.delete("/documents/..%2Fx.pdf").status_code in (400, 404)

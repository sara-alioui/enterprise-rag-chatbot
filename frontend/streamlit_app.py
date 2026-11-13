"""Interface web du chatbot (Streamlit). Elle parle à l'API FastAPI.

Lancement (l'API doit tourner sur le port 8000) :
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os
from urllib.parse import quote

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
LONG_TIMEOUT = 900   # le modèle local peut mettre plusieurs minutes à répondre
ALL_DOCS = "Tous les documents"

st.set_page_config(page_title="Enterprise RAG Chatbot", page_icon="📄")


# ---------- accès à l'API ----------
def api(method: str, path: str, timeout: float = 30, **kwargs):
    """Appelle l'API. Retourne (données, message_d_erreur)."""
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except httpx.TransportError:
        return None, ("L'API ne répond pas. Dans Ubuntu, lance : "
                      "`uvicorn backend.app.main:app --port 8000`")
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):  # erreur de validation
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        return None, str(detail)
    return response.json(), None


# ---------- affichage d'une réponse ----------
ORIGIN_LABELS = {"vector": "similarité", "keyword": "mot-clé", "neighbor": "voisin"}


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)} passages)"):
        for s in sources:
            tag = ORIGIN_LABELS.get(s["origin"], s["origin"])
            if s["origin"] == "vector":
                tag += f" {s['score']:.2f}"
            page = f", page {s['page']}" if s["page"] else ""
            st.markdown(f"**{s['document']}** (passage {s['passage']}{page}, {tag})")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


def show_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("citation"):
        st.markdown(f"> « {msg['citation']} »")
    if msg.get("note"):
        st.caption(msg["note"])
    if msg.get("sources"):
        show_sources(msg["sources"])


# ---------- barre latérale : documents ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("upload_round", 0)

with st.sidebar:
    st.header("Documents")

    health, error = api("GET", "/health")
    if error:
        st.error(error)
    elif not health["llm_reachable"]:
        st.warning("Ollama ne répond pas. Dans Ubuntu, lance : `ollama serve`")

    for text in st.session_state.pop("flash", []):
        (st.warning if text.startswith("⚠") else st.success)(text)

    files = st.file_uploader("Ajouter des PDF, Word ou images",
                             type=["pdf", "docx", "png", "jpg", "jpeg"],
                             accept_multiple_files=True,
                             key=f"uploader{st.session_state.upload_round}")
    if files and st.button("Indexer les fichiers", type="primary"):
        with st.spinner("Indexation en cours (l'OCR peut prendre un moment)..."):
            results, error = api("POST", "/documents", timeout=LONG_TIMEOUT,
                                 files=[("files", (f.name, f.getvalue(), f.type)) for f in files])
        if error:
            st.session_state.flash = [f"⚠ {error}"]
        else:
            st.session_state.flash = [
                (f"{r['name']} : {r['passages']} passages indexés" if r["status"] == "indexé"
                 else f"⚠ {r['name']} : {r['status']}. {r['detail']}") for r in results]
        st.session_state.upload_round += 1
        st.rerun()

    documents, error = api("GET", "/documents")
    documents = documents or []
    st.caption(f"{len(documents)} document(s), {sum(d['passages'] for d in documents)} passages")
    for doc in documents:
        name_col, button_col = st.columns([5, 1])
        name_col.caption(f"📄 {doc['name']} ({doc['passages']})")
        if button_col.button("🗑", key=f"delete_{doc['name']}", help="Supprimer ce document"):
            _, error = api("DELETE", f"/documents/{quote(doc['name'], safe='')}")
            st.session_state.flash = [f"⚠ {error}" if error else f"{doc['name']} supprimé"]
            st.rerun()

    st.divider()
    st.header("Paramètres")
    choice = st.selectbox("Interroger", [ALL_DOCS] + [d["name"] for d in documents])
    k = st.slider("Passages les plus proches", 1, 5, 3)
    neighbors = st.slider("Voisins de chaque passage", 0, 2, 1)
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- conversation ----------
st.title("📄 Chatbot sur mes documents")
st.caption("Ajoute des documents dans la barre latérale, puis pose tes questions.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show_message(msg)

if question := st.chat_input("Ta question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not documents:
            reply = {"role": "assistant",
                     "content": "Aucun document n'est indexé. Ajoute un fichier dans la barre latérale."}
        else:
            body = {"question": question, "k": k, "neighbors": neighbors}
            if choice != ALL_DOCS:
                body["document"] = choice
            with st.spinner("Recherche et génération (peut prendre une minute)..."):
                data, error = api("POST", "/chat", timeout=LONG_TIMEOUT, json=body)
            if error:
                reply = {"role": "assistant", "content": f"⚠ {error}"}
            else:
                reply = {"role": "assistant", "content": data["answer"],
                         "citation": data["citation"], "note": data["note"],
                         "sources": data["sources"] if data["answered"] else []}
        show_message(reply)
    st.session_state.messages.append(reply)

"""Pipeline RAG complet : recherche -> modèle de langage -> vérification."""
from __future__ import annotations

from backend.app.core.logging import get_logger
from backend.app.database.models import Answer
from backend.app.llm.llm_client import OllamaClient
from backend.app.llm.prompt_templates import (ANSWER_SCHEMA, NO_ANSWER, SYSTEM_PROMPT,
                                              build_user_prompt, format_context)
from backend.app.rag.answer_verifier import verify_answer
from backend.app.rag.evidence import select_evidence
from backend.app.retrieval.retriever import HybridRetriever

log = get_logger(__name__)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: OllamaClient):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, k: int | None = None, neighbors: int | None = None,
            source: str | None = None) -> Answer:
        passages = self.retriever.retrieve(question, k, neighbors, source)
        if not passages:
            return Answer(False, NO_ANSWER, note="Aucun document indexé.")
        context = format_context(passages)
        raw = self.llm.chat(SYSTEM_PROMPT, build_user_prompt(question, context),
                            json_schema=ANSWER_SCHEMA)
        check = verify_answer(raw, context)
        if check.ok:
            evidence = select_evidence(passages, check.citation, check.answer)
            return Answer(True, check.answer, check.citation, evidence)
        log.info("réponse écartée (%s) : %s", check.reason, check.unsupported)
        note = ""
        if check.unsupported:
            note = "Réponse écartée : " + ", ".join(f"« {w} »" for w in check.unsupported) \
                   + " n'apparaît pas dans les extraits."
        return Answer(False, NO_ANSWER, note=note)

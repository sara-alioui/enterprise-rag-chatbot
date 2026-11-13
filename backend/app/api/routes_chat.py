"""Poser une question aux documents."""
from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import get_services
from backend.app.api.schemas import ChatRequest, ChatResponse, SourceOut
from backend.app.container import Services
from backend.app.llm.llm_client import LLMUnavailable

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, services: Services = Depends(get_services)) -> ChatResponse:
    if body.document and body.document not in services.store.list_documents():
        raise HTTPException(404, f"Document inconnu : {body.document}")
    try:
        answer = services.rag.ask(body.question, body.k, body.neighbors, body.document)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return ChatResponse(
        answered=answer.answered, answer=answer.text, citation=answer.citation,
        note=answer.note,
        sources=[SourceOut(document=r.chunk.source, passage=r.chunk.index, page=r.chunk.page,
                           origin=r.origin, score=round(r.score, 4), text=r.chunk.text)
                 for r in answer.sources])

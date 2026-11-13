"""Application FastAPI.

Lancement (depuis la racine du projet) :
    uvicorn backend.app.main:app --port 8000
Documentation interactive : http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import routes_chat, routes_documents, routes_health
from backend.app.container import Services, build_services
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, setup_logging

log = get_logger(__name__)


def create_app(services: Services | None = None) -> FastAPI:
    """Fabrique l'application. `services` permet d'injecter de faux composants (tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.services is None:
            setup_logging("INFO")
            app.state.services = build_services()
            log.info("Services prêts : %s passages indexés", app.state.services.store.count())
        yield

    app = FastAPI(title="Enterprise RAG Chatbot",
                  description="Pose des questions à tes documents (PDF, Word, images).",
                  version="0.2.0", lifespan=lifespan)
    app.state.services = services
    app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins),
                       allow_methods=["*"], allow_headers=["*"])
    for module in (routes_health, routes_chat, routes_documents):
        app.include_router(module.router)
    return app


app = create_app()

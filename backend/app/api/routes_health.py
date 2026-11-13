"""État du service."""
from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_services
from backend.app.api.schemas import HealthOut
from backend.app.container import Services

router = APIRouter(tags=["santé"])


@router.get("/health", response_model=HealthOut)
def health(services: Services = Depends(get_services)) -> HealthOut:
    documents = services.store.list_documents()
    llm_ok = services.rag.llm.is_available()
    return HealthOut(status="ok" if llm_ok else "dégradé", documents=len(documents),
                     passages=sum(documents.values()), llm_model=services.settings.llm_model,
                     llm_reachable=llm_ok)

"""Importer, lister et supprimer des documents."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.api.dependencies import get_services
from backend.app.api.schemas import DocumentOut, UploadResult
from backend.app.container import Services
from backend.app.core.logging import get_logger
from backend.app.ingestion.document_loader import SUPPORTED_EXTENSIONS, is_supported

router = APIRouter(tags=["documents"])
log = get_logger(__name__)
_write_lock = threading.Lock()  # une seule écriture à la fois dans la base


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(services: Services = Depends(get_services)) -> list[DocumentOut]:
    docs = services.ingestion.list_documents()
    return [DocumentOut(name=n, passages=c) for n, c in sorted(docs.items())]


@router.post("/documents", response_model=list[UploadResult])
def upload_documents(files: list[UploadFile] = File(...),
                     services: Services = Depends(get_services)) -> list[UploadResult]:
    settings = services.settings
    limit = settings.max_upload_mb * 1024 * 1024
    results: list[UploadResult] = []
    for upload in files:
        name = Path(upload.filename or "").name  # supprime tout chemin éventuel
        if not name or not is_supported(Path(name)):
            results.append(UploadResult(
                name=name or "?", status="refusé",
                detail="Format non pris en charge (" + ", ".join(sorted(SUPPORTED_EXTENSIONS)) + ")."))
            continue
        data = upload.file.read(limit + 1)
        if len(data) > limit:
            results.append(UploadResult(name=name, status="refusé",
                                        detail=f"Fichier trop gros (max {settings.max_upload_mb} Mo)."))
            continue
        destination = settings.docs_dir / name
        with _write_lock:
            try:
                settings.docs_dir.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                passages = services.ingestion.ingest_file(destination)
            except Exception as exc:  # fichier corrompu, etc.
                log.exception("échec d'indexation de %s", name)
                destination.unlink(missing_ok=True)
                results.append(UploadResult(name=name, status="erreur", detail=str(exc)))
                continue
        results.append(UploadResult(
            name=name, status="indexé" if passages else "vide", passages=passages,
            detail="" if passages else "Aucun texte trouvé dans ce fichier."))
    return results


@router.delete("/documents/{name}")
def delete_document(name: str, services: Services = Depends(get_services)) -> dict:
    if Path(name).name != name:
        raise HTTPException(400, "Nom de document invalide.")
    known = name in services.ingestion.list_documents()
    if not known and not (services.settings.docs_dir / name).exists():
        raise HTTPException(404, f"Document inconnu : {name}")
    with _write_lock:
        services.ingestion.remove_document(name)
    return {"deleted": name}

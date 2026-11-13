"""Suivi des fichiers déjà indexés (empreintes), pour l'ingestion incrémentale."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class IngestionState:
    """Mémorise l'empreinte SHA-256 de chaque fichier indexé (fichier JSON)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get(self, name: str) -> str | None:
        return self._data.get(name)

    def set(self, name: str, digest: str) -> None:
        self._data[name] = digest
        self._save()

    def remove(self, name: str) -> None:
        if self._data.pop(name, None) is not None:
            self._save()

    def names(self) -> list[str]:
        return list(self._data)

    @staticmethod
    def file_hash(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

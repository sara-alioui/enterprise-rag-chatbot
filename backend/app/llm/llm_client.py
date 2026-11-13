"""Client du modèle de langage local (Ollama)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from backend.app.core.config import Settings


class LLMUnavailable(RuntimeError):
    """Ollama ne répond pas (serveur arrêté ou modèle absent)."""


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def is_available(self) -> bool:
        """True si le serveur Ollama répond."""
        try:
            with urllib.request.urlopen(f"{self.settings.ollama_url}/api/tags", timeout=2):
                return True
        except Exception:
            return False

    def chat(self, system: str, user: str, json_schema: dict | None = None) -> str:
        payload = {
            "model": self.settings.llm_model,
            "stream": False,
            "options": {"temperature": self.settings.llm_temperature,
                        "num_ctx": self.settings.llm_num_ctx},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if json_schema:
            payload["format"] = json_schema
        request = urllib.request.Request(
            f"{self.settings.ollama_url}/api/chat", json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout) as resp:
                return json.load(resp)["message"]["content"]
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise LLMUnavailable(
                "Ollama ne répond pas. Lance `ollama serve` dans Ubuntu.") from exc

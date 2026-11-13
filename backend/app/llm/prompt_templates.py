"""Consignes envoyées au modèle de langage."""
from __future__ import annotations

from backend.app.database.models import RetrievedChunk

NO_ANSWER = "Je ne trouve pas la réponse dans les documents."

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond en français à partir uniquement des extraits fournis. "
    "N'invente rien : n'ajoute aucune information absente des extraits. "
    "Réponds en 2 à 5 phrases, en reprenant les détails précis des extraits "
    "(noms, conditions, valeurs, coefficients). Si les extraits ne contiennent pas la réponse, "
    f"réponds exactement : « {NO_ANSWER} »"
)

# Le modèle doit répondre en JSON : une citation copiée des extraits + la réponse.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"citation": {"type": "string"}, "reponse": {"type": "string"}},
    "required": ["citation", "reponse"],
}


def format_context(passages: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Extrait {n} - {p.chunk.source}]\n{p.chunk.text}"
                       for n, p in enumerate(passages, start=1))


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"Extraits :\n{context}\n\nQuestion : {question}\n\n"
        "Réponds en JSON avec deux champs : « citation » = une phrase copiée mot pour mot "
        "depuis les extraits et qui contient la réponse (chaîne vide si aucun extrait ne "
        "contient la réponse), « reponse » = ta réponse en français fondée sur cette citation."
    )

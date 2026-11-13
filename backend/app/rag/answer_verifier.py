"""Contrôle anti-invention : la réponse doit s'appuyer sur les extraits fournis."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from backend.app.llm.prompt_templates import NO_ANSWER


@dataclass
class Verification:
    ok: bool
    reason: str = ""            # "invalid_json" | "no_citation" | "citation_not_in_context" | "unsupported_terms"
    answer: str = ""
    citation: str = ""
    unsupported: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Minuscules sans accents."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", _normalize(text)))


def parse_llm_json(raw: str) -> tuple[str, str] | None:
    """(citation, réponse) extraits de la sortie JSON du modèle, ou None."""
    data = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
    if not isinstance(data, dict):
        return None
    return str(data.get("citation", "")), str(data.get("reponse", ""))


def verify_answer(raw: str, context: str) -> Verification:
    """Accepte la réponse si :
    1. la citation fournie figure (à 80 % au moins) dans les extraits ;
    2. les noms propres et les chiffres de la réponse figurent aussi dans les extraits.
    """
    parsed = parse_llm_json(raw)
    if parsed is None:
        return Verification(False, "invalid_json")
    citation, answer = parsed
    answer = answer.strip()
    context_tokens = _tokens(context)

    words = [w for w in re.findall(r"\w+", _normalize(citation)) if len(w) >= 5]
    if not words or not answer or NO_ANSWER in answer:
        return Verification(False, "no_citation")
    if sum(w in context_tokens for w in words) / len(words) < 0.8:
        return Verification(False, "citation_not_in_context")

    citation_tokens = _tokens(citation)
    unsupported: list[str] = []
    for m in re.finditer(r"\w+", answer):
        word = m.group()
        before = answer[:m.start()].rstrip()
        sentence_start = before == "" or before[-1] in ".?!:\n"
        capitalized = word[0].isupper() and len(word) >= 3 and not sentence_start
        has_digit = any(ch.isdigit() for ch in word)
        norm = _normalize(word)
        # un chiffre doit figurer dans la citation elle-même (pas seulement dans le contexte)
        reference = citation_tokens if has_digit else context_tokens
        if (capitalized or has_digit) and norm not in reference and word not in unsupported:
            unsupported.append(word)
    if unsupported:
        return Verification(False, "unsupported_terms", answer, citation.strip(), unsupported)
    return Verification(True, "", answer, citation.strip())

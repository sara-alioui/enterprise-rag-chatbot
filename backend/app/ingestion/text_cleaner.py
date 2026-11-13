"""Nettoyage du texte extrait des documents."""
import re

# mot coupé en fin de ligne : « Appli-\nquées » -> « Appliquées »
_HYPHENATION = re.compile(r"(?<=[a-zà-öø-ÿ])-\n(?=[a-zà-öø-ÿ])")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = _HYPHENATION.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

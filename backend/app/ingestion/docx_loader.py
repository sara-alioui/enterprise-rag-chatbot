"""Lecture des fichiers Word (.docx) : paragraphes et tableaux."""
from pathlib import Path

from docx import Document

from backend.app.database.models import Page


def load_docx(path: Path) -> list[Page]:
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return [Page(0, "\n".join(lines))]

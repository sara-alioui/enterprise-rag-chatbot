import io

import pymupdf
import pytest
from docx import Document

from backend.app.chunking.text_chunker import chunk_text
from backend.app.ingestion.document_loader import is_supported, load_document
from backend.app.ingestion.ocr import tesseract_available
from backend.app.ingestion.text_cleaner import clean_text


# ---------- nettoyage et découpage ----------
def test_clean_text_joins_hyphenated_words():
    assert clean_text("Sciences Appli-\nquées") == "Sciences Appliquées"


def test_clean_text_keeps_real_hyphens_and_code():
    assert clean_text("a - b\nx = y-1") == "a - b\nx = y-1"


def test_chunk_text_never_cuts_a_line_and_overlaps():
    lines = [f"ligne numero {i} avec quelques mots de plus" for i in range(12)]
    chunks = chunk_text("\n".join(lines), max_words=20, min_words=3, overlap_lines=1)
    assert len(chunks) > 1
    for c in chunks:
        assert all(line in lines for line in c.split("\n"))
    assert chunks[0].split("\n")[-1] == chunks[1].split("\n")[0]  # chevauchement


def test_chunk_text_drops_tiny_chunks():
    assert chunk_text("trop court", max_words=20, min_words=8) == []


# ---------- lecture des formats ----------
def test_load_docx_reads_paragraphs_and_tables(tmp_path, settings):
    doc = Document()
    doc.add_paragraph("Premier paragraphe du rapport.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Projet"
    table.rows[0].cells[1].text = "Atlas"
    path = tmp_path / "a.docx"
    doc.save(path)
    text = load_document(path, settings)[0].text
    assert "Premier paragraphe" in text and "Projet | Atlas" in text


def test_load_pdf_reads_text_per_page(tmp_path, settings):
    pdf = pymupdf.open()
    for n in (1, 2):
        pdf.new_page().insert_text((72, 100), f"Contenu de la page {n}", fontsize=12)
    path = tmp_path / "a.pdf"
    pdf.save(path)
    pages = load_document(path, settings)
    assert [p.number for p in pages] == [1, 2]
    assert "page 2" in pages[1].text


def test_unsupported_format(tmp_path, settings):
    assert not is_supported(tmp_path / "a.txt")
    with pytest.raises(ValueError):
        load_document(tmp_path / "a.txt", settings)


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract non installé")
def test_ocr_reads_scanned_pdf_and_image(tmp_path, settings):
    from dataclasses import replace
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    img = Image.new("RGB", (1000, 200), "white")
    ImageDraw.Draw(img).text((30, 60), "Montant total : 4 250 dirhams", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    png = tmp_path / "facture.png"
    png.write_bytes(buf.getvalue())
    pdf = pymupdf.open()
    pdf.new_page().insert_image(pymupdf.Rect(0, 0, 595, 120), stream=buf.getvalue())
    scan = tmp_path / "scan.pdf"
    pdf.save(scan)
    ocr_settings = replace(settings, ocr_enabled=True)
    assert "dirhams" in load_document(png, ocr_settings)[0].text
    assert "dirhams" in load_document(scan, ocr_settings)[0].text


# ---------- service d'ingestion ----------
def _write_docx(path, text):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(path)


def test_sync_is_incremental(settings, ingestion):
    _write_docx(settings.docs_dir / "a.docx", "Le chat dort sur le canapé du salon toute la journée.\n" * 3)
    first = ingestion.sync_folder()
    assert first[0].status == "indexé" and first[0].passages > 0
    assert ingestion.sync_folder()[0].status == "inchangé"           # rien à refaire

    _write_docx(settings.docs_dir / "a.docx", "Un texte entièrement différent, sur la cuisine et les recettes.\n" * 3)
    assert ingestion.sync_folder()[0].status == "indexé"             # modifié -> réindexé

    (settings.docs_dir / "a.docx").unlink()
    assert ingestion.sync_folder()[0].status == "supprimé"           # supprimé -> retiré
    assert ingestion.list_documents() == {}


def test_remove_document_deletes_file_and_passages(settings, ingestion):
    path = settings.docs_dir / "b.docx"
    _write_docx(path, "Des phrases suffisamment longues pour former un passage complet ici.\n" * 2)
    ingestion.ingest_file(path)
    assert "b.docx" in ingestion.list_documents()
    ingestion.remove_document("b.docx")
    assert ingestion.list_documents() == {} and not path.exists()


def test_reingesting_same_name_replaces_passages(settings, ingestion):
    path = settings.docs_dir / "c.docx"
    _write_docx(path, "Première version avec assez de mots pour un passage correct.\n" * 2)
    ingestion.ingest_file(path)
    _write_docx(path, "Seconde version avec assez de mots pour un passage correct.\n" * 2)
    ingestion.ingest_file(path)
    docs = ingestion.list_documents()
    assert list(docs) == ["c.docx"] and docs["c.docx"] == 1

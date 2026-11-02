"""Step 1: load documents (PDF, DOCX, TXT, MD) -> list of (page_number, text)."""
import os
from typing import List, Tuple


def load_document(path: str) -> List[Tuple[int, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]
    raise ValueError(f"Unsupported file type: {ext}")


def _load_pdf(path: str) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def _load_docx(path: str) -> List[Tuple[int, str]]:
    from docx import Document

    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(1, text)]

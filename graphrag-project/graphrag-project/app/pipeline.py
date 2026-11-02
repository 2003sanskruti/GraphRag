"""Ingestion pipeline: load -> chunk -> embed -> extract graph -> store in Neo4j."""
import hashlib
import os
from typing import Dict

from .chunker import chunk_pages
from .embedder import embed_texts
from .extractor import extract_all
from .graph_store import GraphStore
from .loader import load_document


def ingest_file(path: str, store: GraphStore, build_graph: bool = True) -> Dict:
    name = os.path.basename(path)
    with open(path, "rb") as f:
        doc_id = hashlib.sha1(f.read()).hexdigest()[:16]

    pages = load_document(path)
    if not pages:
        raise ValueError("No extractable text found (scanned PDF? try OCR first).")

    chunks = chunk_pages(pages)
    for c in chunks:
        c["id"] = f"{doc_id}-{c['index']}"

    vectors = embed_texts([c["text"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["embedding"] = v

    store.add_document(doc_id, name, chunks)

    entities = relations = 0
    if build_graph:
        extractions = extract_all(chunks)
        store.add_graph(extractions)
        entities = sum(len(x["entities"]) for x in extractions)
        relations = sum(len(x["relations"]) for x in extractions)

    return {"doc_id": doc_id, "name": name, "pages": len(pages), "chunks": len(chunks),
            "entities": entities, "relations": relations}

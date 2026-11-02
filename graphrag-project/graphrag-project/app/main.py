"""FastAPI server: upload documents, ask questions. Run: uvicorn app.main:app --reload"""
import truststore

truststore.inject_into_ssl()

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import UPLOAD_DIR
from .graph_store import GraphStore
from .pipeline import ingest_file
from .rag import answer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="GraphRAG with Neo4j")
store = GraphStore()


@app.on_event("startup")
def _startup():
    store.setup()


@app.on_event("shutdown")
def _shutdown():
    store.close()


class Ask(BaseModel):
    question: str
    k: int = 5


@app.post("/upload")
def upload(file: UploadFile = File(...), build_graph: bool = True):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt", ".md"):
        raise HTTPException(400, "Supported: pdf, docx, txt, md")
    path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return ingest_file(path, store, build_graph=build_graph)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ask")
def ask(body: Ask):
    return answer(body.question, store, k=body.k)


@app.get("/documents")
def documents():
    return store.list_documents()


@app.delete("/documents/{doc_id}")
def delete(doc_id: str):
    store.delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/stats")
def stats():
    return store.stats()


app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))

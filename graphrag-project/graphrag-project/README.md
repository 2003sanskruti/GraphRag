# GraphRAG with Neo4j (no Docker)

Pipeline: **PDF/DOCX/TXT → chunk → embeddings → Groq entity/relation extraction → Neo4j (graph + vector index) → GraphRAG Q&A**

## Architecture
```
Upload ─► loader.py ─► chunker.py ─► embedder.py ─► graph_store.py (Chunk + vector index)
                                  └► extractor.py (Groq) ─► Entity / RELATED edges
Question ─► embed ─► vector search (top-k chunks)
          ─► graph expansion (entity relations + chunks sharing entities)
          ─► Groq answers with passages + graph facts + citations
```
Graph model:
```
(Document)-[:HAS_CHUNK]->(Chunk)-[:NEXT]->(Chunk)
(Chunk)-[:MENTIONS]->(Entity)-[:RELATED {type}]->(Entity)
```

## Setup
1. Install Neo4j **without Docker** — pick one:
   - Neo4j Desktop (https://neo4j.com/download/) -> create a local DBMS (v5.18+), start it, set a password
   - Neo4j Aura Free (cloud) -> use the `neo4j+s://...` URI it gives you
2. Create a Groq API key at [console.groq.com/keys](https://console.groq.com/keys). Add it to the project `.env` file as `GROQ_API_KEY=your-key` (no quotes). Groq's free developer access has usage limits that may vary by account and model. The default model is `qwen/qwen3.8-27b`; set `GROQ_MODEL` in `.env` only if you want to use another model enabled for your account.
3. Python 3.10+ and project dependencies. On Windows, from the project directory:
   ```bash
   py -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   Configure your Neo4j connection in `.env` as well. This app uses Groq for both entity extraction and question answering.
4. Run the web app:
   ```bash
   .venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```
   Open http://127.0.0.1:8000 -> upload a PDF -> ask questions.

## CLI alternative
```bash
python cli.py ingest data/yourfile.pdf
python cli.py ask "What is the main topic of the document?"
python cli.py stats
```
Use `--no-graph` on ingest to skip entity extraction (plain vector RAG, faster).

## Explore the graph in Neo4j Browser
```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) RETURN d,c,e LIMIT 50;
MATCH (a:Entity)-[r:RELATED]->(b:Entity) RETURN a,r,b LIMIT 50;
```

## Interview talking points
- **Why GraphRAG?** Plain vector RAG finds similar chunks; the graph adds multi-hop links between entities across chunks and documents.
- **Hybrid retrieval:** vector top-k, then expand via shared entities (`MENTIONS`) and entity relations (`RELATED`).
- **Why local embeddings?** Free, no extra API key; change `EMBED_MODEL` / `EMBED_DIM` to swap.
- **Improvements:** entity resolution/dedup, community detection + summaries (Microsoft GraphRAG), Cypher-generating agent, reranking, OCR for scanned PDFs, background ingestion.

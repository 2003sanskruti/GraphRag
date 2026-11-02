"""Query time: vector search -> graph expansion -> Claude answer (GraphRAG)."""
from typing import Dict

from .embedder import embed_query
from .graph_store import GraphStore
from .llm import ask_claude

SYSTEM = """You answer questions using ONLY the provided context (text passages and knowledge-graph facts).
If the answer is not in the context, say you don't know. Cite passages like [1], [2]."""


def answer(question: str, store: GraphStore, k: int = 5) -> Dict:
    hits = store.vector_search(embed_query(question), k=k)
    if not hits:
        return {"answer": "No documents indexed yet.", "sources": [], "facts": []}

    triples, related = store.graph_context([h["id"] for h in hits])
    passages = hits + related

    ctx = "PASSAGES:\n" + "\n\n".join(
        f"[{i}] ({p.get('doc')}, p.{p.get('page')}) {p['text']}" for i, p in enumerate(passages, 1)
    )
    if triples:
        ctx += "\n\nGRAPH FACTS:\n" + "\n".join(
            f"- {t['source']} -[{t['rel']}]-> {t['target']}" for t in triples
        )

    reply = ask_claude(SYSTEM, f"{ctx}\n\nQUESTION: {question}")
    return {
        "answer": reply,
        "sources": [{"n": i, "doc": p.get("doc"), "page": p.get("page"),
                     "score": round(p["score"], 3) if "score" in p else None,
                     "via": "vector" if "score" in p else "graph",
                     "preview": p["text"][:200]} for i, p in enumerate(passages, 1)],
        "facts": [f"{t['source']} -[{t['rel']}]-> {t['target']}" for t in triples],
    }

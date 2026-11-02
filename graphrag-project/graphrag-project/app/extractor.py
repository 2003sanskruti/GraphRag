"""Step 4: extract entities + relations from each chunk with Groq (the 'Graph' in GraphRAG)."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .llm import ask_llm

SYSTEM = """You extract a knowledge graph from text.
Return ONLY valid JSON, no markdown, in this exact shape:
{"entities":[{"name":"...","type":"PERSON|ORG|CONCEPT|TECH|LOCATION|EVENT|OTHER"}],
 "relations":[{"source":"...","target":"...","type":"SHORT_VERB_PHRASE_UPPER_SNAKE"}]}
Rules: max 12 entities, max 12 relations, use canonical names, relation source/target must appear in entities."""


def _parse(raw: str) -> Dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {"entities": [], "relations": []}


def extract_chunk(chunk_id: str, text: str) -> Dict:
    try:
        data = _parse(ask_llm(SYSTEM, f"TEXT:\n{text}", max_tokens=1200, json_mode=True))
    except Exception as e:  # keep ingestion going even if one chunk fails
        print(f"[extract] chunk {chunk_id} failed: {e}")
        data = {"entities": [], "relations": []}
    return {"chunk_id": chunk_id,
            "entities": data.get("entities", []),
            "relations": data.get("relations", [])}


def extract_all(chunks: List[Dict], workers: int = 4) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda c: extract_chunk(c["id"], c["text"]), chunks))

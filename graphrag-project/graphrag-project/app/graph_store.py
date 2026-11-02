"""Step 5: store Documents / Chunks / Entities in Neo4j + vector index.

Graph model
  (Document)-[:HAS_CHUNK]->(Chunk)-[:NEXT]->(Chunk)
  (Chunk)-[:MENTIONS]->(Entity)-[:RELATED {type}]->(Entity)
"""
from typing import Dict, List

from neo4j import GraphDatabase

from .config import EMBED_DIM, NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def _run(self, query: str, **params):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            return [r.data() for r in s.run(query, **params)]

    # ---------- schema ----------
    def setup(self):
        self.driver.verify_connectivity()
        self._run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        self._run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self._run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
        self._run(
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
            dim=EMBED_DIM,
        )

    # ---------- write ----------
    def add_document(self, doc_id: str, name: str, chunks: List[Dict]):
        self._run(
            """
            MERGE (d:Document {id:$doc_id}) SET d.name=$name, d.created=datetime()
            WITH d UNWIND $chunks AS ch
            MERGE (c:Chunk {id: ch.id})
            SET c.text=ch.text, c.index=ch.index, c.page=ch.page, c.embedding=ch.embedding
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            doc_id=doc_id, name=name, chunks=chunks,
        )
        self._run(
            """
            MATCH (d:Document {id:$doc_id})-[:HAS_CHUNK]->(a:Chunk),
                  (d)-[:HAS_CHUNK]->(b:Chunk)
            WHERE b.index = a.index + 1
            MERGE (a)-[:NEXT]->(b)
            """,
            doc_id=doc_id,
        )

    def add_graph(self, extractions: List[Dict]):
        ents, rels = [], []
        for ex in extractions:
            for e in ex["entities"]:
                name = (e.get("name") or "").strip()
                if name:
                    ents.append({"chunk_id": ex["chunk_id"], "name": name,
                                 "key": name.lower(), "type": e.get("type", "OTHER")})
            for r in ex["relations"]:
                s, t = (r.get("source") or "").strip(), (r.get("target") or "").strip()
                if s and t:
                    rels.append({"chunk_id": ex["chunk_id"], "source": s.lower(), "target": t.lower(),
                                 "source_name": s, "target_name": t,
                                 "type": (r.get("type") or "RELATED_TO").upper()})
        if ents:
            self._run(
                """
                UNWIND $items AS e
                MATCH (c:Chunk {id:e.chunk_id})
                MERGE (n:Entity {key:e.key}) ON CREATE SET n.name=e.name, n.type=e.type
                MERGE (c)-[:MENTIONS]->(n)
                """,
                items=ents,
            )
        if rels:
            self._run(
                """
                UNWIND $rels AS r
                MERGE (a:Entity {key:r.source}) ON CREATE SET a.name=r.source_name
                MERGE (b:Entity {key:r.target}) ON CREATE SET b.name=r.target_name
                MERGE (a)-[x:RELATED {type:r.type}]->(b)
                SET x.chunk_id = r.chunk_id
                """,
                rels=rels,
            )

    # ---------- read ----------
    def vector_search(self, embedding: List[float], k: int = 5) -> List[Dict]:
        return self._run(
            """
            CALL db.index.vector.queryNodes('chunk_embedding', $k, $emb) YIELD node, score
            OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(node)
            RETURN node.id AS id, node.text AS text, node.page AS page,
                   d.name AS doc, score
            ORDER BY score DESC
            """,
            k=k, emb=embedding,
        )

    def graph_context(self, chunk_ids: List[str], extra_chunks: int = 3, max_rels: int = 40):
        triples = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.id IN $ids
            MATCH (e)-[r:RELATED]-(o:Entity)
            RETURN DISTINCT
              CASE WHEN startNode(r)=e THEN e.name ELSE o.name END AS source,
              r.type AS rel,
              CASE WHEN startNode(r)=e THEN o.name ELSE e.name END AS target
            LIMIT $n
            """,
            ids=chunk_ids, n=max_rels,
        )
        related = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(o:Chunk)
            WHERE c.id IN $ids AND NOT o.id IN $ids
            OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(o)
            RETURN o.id AS id, o.text AS text, o.page AS page, d.name AS doc,
                   count(DISTINCT e) AS shared
            ORDER BY shared DESC LIMIT $n
            """,
            ids=chunk_ids, n=extra_chunks,
        )
        return triples, related

    def list_documents(self) -> List[Dict]:
        return self._run(
            """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.id AS id, d.name AS name, count(c) AS chunks, toString(d.created) AS created
            ORDER BY created DESC
            """
        )

    def delete_document(self, doc_id: str):
        self._run("MATCH (d:Document {id:$id})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c", id=doc_id)
        self._run("MATCH (d:Document {id:$id}) DETACH DELETE d", id=doc_id)
        self._run("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e")

    def stats(self) -> Dict:
        rows = self._run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"
        )
        return {r["label"]: r["n"] for r in rows}

"""CLI usage:
  python cli.py ingest data/sample.pdf [more files...] [--no-graph]
  python cli.py ask "What is ...?"
  python cli.py stats
"""
import json
import sys

from app.graph_store import GraphStore
from app.pipeline import ingest_file
from app.rag import answer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    store = GraphStore()
    store.setup()
    cmd = sys.argv[1]
    try:
        if cmd == "ingest":
            no_graph = "--no-graph" in sys.argv
            for p in [a for a in sys.argv[2:] if not a.startswith("--")]:
                print(json.dumps(ingest_file(p, store, build_graph=not no_graph), indent=2))
        elif cmd == "ask":
            r = answer(" ".join(sys.argv[2:]), store)
            print("\nANSWER:\n", r["answer"])
            print("\nGRAPH FACTS:", *r["facts"][:10], sep="\n  ")
            print("\nSOURCES:", *[f"[{s['n']}] {s['doc']} p.{s['page']} ({s['via']})" for s in r["sources"]], sep="\n  ")
        elif cmd == "stats":
            print(store.stats())
        else:
            print(__doc__)
    finally:
        store.close()


if __name__ == "__main__":
    main()

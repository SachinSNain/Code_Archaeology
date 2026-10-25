"""One-shot environment verification + schema bootstrap.

Run after `docker compose up -d` to: create Postgres tables, create Neo4j
constraints/indexes, and confirm Chroma is reachable. Safe to re-run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `src` is importable

from src.db import chroma, neo4j, postgres


def main() -> None:
    print("== PostgreSQL ==")
    postgres.init_db()
    with postgres.get_engine().connect() as conn:
        pass
    print("  tables created/verified:", list(postgres.Base.metadata.tables.keys()))

    print("== Neo4j ==")
    neo4j.init_constraints()
    with neo4j.session() as s:
        s.run("RETURN 1")
    print("  constraints created/verified")
    neo4j.close_driver()

    print("== Chroma ==")
    hb = chroma.heartbeat()
    print(f"  heartbeat OK ({hb})")

    print("\nAll three stores are reachable and schema is initialized.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface a clear failure for whichever store isn't ready
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

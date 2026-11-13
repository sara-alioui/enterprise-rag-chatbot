"""Outil en ligne de commande, pour contrôler le projet sans interface.

Depuis la racine du projet :
    python -m backend.app.cli sync                 # indexe le dossier docs/
    python -m backend.app.cli docs                 # liste les documents indexés
    python -m backend.app.cli ask "Ta question ?"  # pose une question
    python -m backend.app.cli ask "..." --doc fichier.pdf
    python -m backend.app.cli remove fichier.pdf
"""
from __future__ import annotations

import argparse

from backend.app.container import build_services
from backend.app.core.logging import setup_logging
from backend.app.llm.llm_client import LLMUnavailable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backend.app.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="indexer le dossier docs/")
    sub.add_parser("docs", help="lister les documents indexés")
    ask = sub.add_parser("ask", help="poser une question")
    ask.add_argument("question")
    ask.add_argument("--doc", default=None, help="limiter la recherche à ce document")
    rm = sub.add_parser("remove", help="supprimer un document")
    rm.add_argument("name")
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    services = build_services()

    if args.command == "sync":
        for r in services.ingestion.sync_folder():
            print(f"{r.name}: {r.status}" + (f" ({r.passages} passages)" if r.passages else ""))
        print(f"Base : {services.store.count()} passages")
    elif args.command == "docs":
        for name, n in sorted(services.ingestion.list_documents().items()):
            print(f"{name}: {n} passages")
    elif args.command == "remove":
        services.ingestion.remove_document(args.name)
        print(f"{args.name}: supprimé")
    elif args.command == "ask":
        try:
            answer = services.rag.ask(args.question, source=args.doc)
        except LLMUnavailable as exc:
            print(exc)
            return 1
        print(answer.text)
        if answer.citation:
            print(f"\n> « {answer.citation} »")
        if answer.note:
            print(f"\n({answer.note})")
        for p in answer.sources:
            tag = {"vector": f"similarité {p.score:.2f}", "keyword": "mot-clé",
                   "neighbor": "voisin"}[p.origin]
            page = f", page {p.chunk.page}" if p.chunk.page else ""
            print(f"  - {p.chunk.source} (passage {p.chunk.index}{page}, {tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

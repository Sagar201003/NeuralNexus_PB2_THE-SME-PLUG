"""
main.py
-------
SME-PLUG entry point.

Usage:
    python main.py demo                          — Run end-to-end demo
    python main.py api                           — Start FastAPI server
    python main.py ingest --domain <domain_id>   — Ingest docs for a capsule
    python main.py capsule create --domain "Name" — Create new capsule
    python main.py query "Your question here"    — One-shot query
"""

import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()


def run_demo():
    from demo.run_demo import run_demo
    run_demo()


def run_api():
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
    )


def run_ingest(domain_id: str):
    from cli.capsule_creator import ingest_capsule
    ingest_capsule(domain_id)


def run_capsule_create(domain: str, docs: str = None):
    from cli.capsule_creator import create_capsule
    create_capsule(domain, docs)


def run_query(query: str, domain: str = None):
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    from core.expert_core import ExpertCore
    from core.confidence_gate import ExpertResponse, ClarificationResponse
    from rag.vector_store import VectorStoreManager
    from rag.ingestion import DocumentIngestionPipeline, BM25Index

    console = Console()
    console.print("[cyan]🔧 Initializing ExpertCore...[/]")

    vector_store = VectorStoreManager()
    bm25_index = BM25Index()

    ec = ExpertCore()
    ec.retriever.set_stores(vector_store, bm25_index)
    ec.bootstrap()

    # Ingest capsule knowledge (skip if already ingested)
    pipeline = DocumentIngestionPipeline(vector_store, bm25_index)
    for did, capsule in ec.router._capsules.items():
        if capsule.knowledge_dir:
            stats = vector_store.collection_stats(did)
            if stats["document_count"] > 0:
                console.print(f"[dim]⏭️  {did}: {stats['document_count']} chunks already indexed, skipping.[/]")
                continue
            pipeline.ingest_capsule(
                domain_id=did,
                knowledge_dir=capsule.knowledge_dir,
                chunk_size=capsule.rag.chunk_size,
                chunk_overlap=capsule.rag.chunk_overlap,
                embedding_model=capsule.rag.embedding_model,
            )

    console.print(f"[cyan]🔍 Querying: {query}[/]\n")

    result = ec.query(query, force_domain=domain)

    if isinstance(result, ClarificationResponse):
        console.print(Panel(f"[yellow]{result.message}[/]", title="❓ Clarification"))
    elif isinstance(result, ExpertResponse):
        console.print(Panel(
            f"[bold]{result.domain_name}[/] | "
            f"Confidence: {result.confidence_score:.2f} | "
            f"Layer: {result.detection_layer}",
            title="🔬 Domain Detected",
            border_style="blue",
        ))
        if result.decision_tree_path:
            console.print(f"🌳 Path: {' → '.join(result.decision_tree_path)}\n")
        console.print(Panel(Markdown(result.answer), title="💡 Expert Answer", border_style="green"))
        if result.guardrail_warnings:
            for w in result.guardrail_warnings:
                console.print(f"⚠️ {w}")


def main():
    parser = argparse.ArgumentParser(
        description="🧬 SME-PLUG: Universal Subject Matter Expert Plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo
  python main.py api
  python main.py query "Is this beam safe for 500 kN?"
  python main.py query "Triage this CVE-2024-3400 alert" --domain cybersecurity
  python main.py ingest --domain structural_engineering
  python main.py capsule create --domain "Petroleum Engineering" --docs ./pdfs/
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo
    subparsers.add_parser("demo", help="Run end-to-end multi-domain demo")

    # API Server
    subparsers.add_parser("api", help="Start FastAPI server")

    # Query
    query_parser = subparsers.add_parser("query", help="One-shot expert query")
    query_parser.add_argument("text", help="Your question or task")
    query_parser.add_argument("--domain", help="Force a specific domain capsule")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest docs for a capsule")
    ingest_parser.add_argument("--domain", required=True, help="Domain ID slug")

    # Capsule management
    capsule_parser = subparsers.add_parser("capsule", help="Capsule management")
    capsule_sub = capsule_parser.add_subparsers(dest="capsule_command")
    create_parser = capsule_sub.add_parser("create", help="Create new capsule")
    create_parser.add_argument("--domain", required=True, help="Domain name")
    create_parser.add_argument("--docs", help="Path to documents directory")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "api":
        run_api()
    elif args.command == "query":
        run_query(args.text, getattr(args, "domain", None))
    elif args.command == "ingest":
        run_ingest(args.domain)
    elif args.command == "capsule" and getattr(args, "capsule_command", None) == "create":
        run_capsule_create(args.domain, getattr(args, "docs", None))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
demo/run_demo.py
----------------
End-to-end demonstration of the SME-PLUG.

Fires one query per domain, showing:
  - Domain detection layer used
  - Context chunks retrieved with citations
  - Decision tree traversal
  - Final expert answer

Run: python demo/run_demo.py
"""

import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from core.expert_core import ExpertCore
from core.confidence_gate import ExpertResponse, ClarificationResponse
from rag.vector_store import VectorStoreManager
from rag.ingestion import DocumentIngestionPipeline, BM25Index
from demo.demo_queries import DEMO_QUERIES

console = Console()


def print_header():
    console.print()
    console.print(Panel(
        "[bold cyan]🧬 SME-PLUG: Universal Subject Matter Expert Plugin[/]\n\n"
        "[dim]Hot-swappable domain expertise with Advanced RAG,\n"
        "decision trees, and source-of-truth citations.[/]",
        border_style="cyan",
        expand=False,
    ))
    console.print()


def print_result(result, domain: str, query_info: dict, elapsed: float):
    """Format and print an expert response."""
    if isinstance(result, ClarificationResponse):
        console.print(Panel(
            f"[yellow]❓ Clarification Needed[/]\n\n{result.message}",
            title=f"🔍 {domain}",
            border_style="yellow",
        ))
        return

    if isinstance(result, ExpertResponse):
        # Detection info
        layer_names = {0: "Forced", 1: "Keyword/Regex", 2: "LLM Classifier", 3: "Embedding Similarity"}
        layer_name = layer_names.get(result.detection_layer, f"Layer {result.detection_layer}")

        # Build info table
        info_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        info_table.add_column("Key", style="bold cyan", width=20)
        info_table.add_column("Value")
        info_table.add_row("🔬 Domain", f"{result.domain_name}")
        info_table.add_row("🎯 Detection Layer", f"{layer_name}")
        info_table.add_row("📊 Confidence", f"{result.confidence_score:.2f}")
        info_table.add_row("⏱️ Response Time", f"{elapsed:.2f}s")
        info_table.add_row("🛡️ Guardrails", "✅ Passed" if result.guardrail_passed else "⚠️ Warnings")

        if result.decision_tree_path:
            path_str = " → ".join(result.decision_tree_path[:6])
            info_table.add_row("🌳 Decision Path", path_str)

        if result.citations:
            sources = set(c.get("source", "") for c in result.citations)
            info_table.add_row("📚 Sources", ", ".join(sources))

        console.print()
        console.print(Panel(
            f"[bold]{query_info['description']}[/]",
            title=f"📋 Query: {domain.replace('_', ' ').title()}",
            border_style="blue",
        ))
        console.print(f"  [dim]Q: {query_info['query'][:120]}...[/]" if len(query_info['query']) > 120 else f"  [dim]Q: {query_info['query']}[/]")
        console.print()
        console.print(info_table)
        console.print()
        console.print(Panel(
            Markdown(result.answer[:2000]),
            title="💡 Expert Answer",
            border_style="green",
        ))

        if result.guardrail_warnings:
            for w in result.guardrail_warnings:
                console.print(f"  [yellow]⚠️ {w}[/]")

        console.print("─" * 80)


def run_demo():
    """Run the full end-to-end demo."""
    print_header()

    # ── Step 1: Initialize ────────────────────────────────────────────────
    console.print("[cyan]🔧 Initializing ExpertCore...[/]")

    vector_store = VectorStoreManager()
    bm25_index = BM25Index()

    ec = ExpertCore()
    ec.retriever.set_stores(vector_store, bm25_index)

    console.print("[cyan]🔄 Bootstrapping domain router...[/]")
    ec.bootstrap()

    # ── Step 2: Ingest knowledge ──────────────────────────────────────────
    console.print("[cyan]📥 Ingesting capsule knowledge bases...[/]")
    pipeline = DocumentIngestionPipeline(vector_store, bm25_index)

    for domain_id, capsule in ec.router._capsules.items():
        if capsule.knowledge_dir:
            stats = vector_store.collection_stats(domain_id)
            if stats["document_count"] > 0:
                console.print(f"  ⏭️  {domain_id}: {stats['document_count']} chunks already indexed, skipping.")
                continue
            summary = pipeline.ingest_capsule(
                domain_id=domain_id,
                knowledge_dir=capsule.knowledge_dir,
                chunk_size=capsule.rag.chunk_size,
                chunk_overlap=capsule.rag.chunk_overlap,
                embedding_model=capsule.rag.embedding_model,
            )
            console.print(f"  ✅ {domain_id}: {summary['chunks_stored']} chunks indexed")

    console.print()
    console.print("[bold green]🚀 Demo ready! Querying all 3 domains...[/]")
    console.print("═" * 80)

    # ── Step 3: Run queries ───────────────────────────────────────────────
    for domain, queries in DEMO_QUERIES.items():
        query_info = queries[0]  # Use first query per domain for demo

        start = time.time()
        result = ec.query(query_info["query"])
        elapsed = time.time() - start

        print_result(result, domain, query_info, elapsed)

    # ── Summary ───────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold green]✅ Demo complete![/]\n\n"
        "Four domains queried with automatic domain detection,\n"
        "advanced RAG retrieval, decision tree reasoning,\n"
        "and source-of-truth citation enforcement.\n\n"
        "[dim]Available capsules:[/]\n"
        "  🏗️ Structural Engineering\n"
        "  🛡️ Cybersecurity SOC Analyst\n"
        "  ⚖️ Legal Contract Analyst\n"
        "  💰 US Tax Expert\n\n"
        "[dim]Run the API server: python main.py api[/]",
        title="🏆 SME-PLUG Demo Summary",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()

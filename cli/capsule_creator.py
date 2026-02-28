"""
capsule_creator.py
------------------
CLI tool for creating new DNA capsules.

Usage:
    python -m cli.capsule_creator --domain "Petroleum Engineering" --docs ./pdfs/
    python -m cli.capsule_creator --domain "Healthcare" --docs ./medical_docs/ --ingest
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()


DEFAULT_CAPSULE_TEMPLATE = {
    "version": "1.0.0",
    "expert_persona": "You are a domain expert with deep knowledge in {domain}.\nYou provide precise, factual answers with source citations.\nWhen uncertain, you clearly state limitations.",
    "triggers": {
        "keywords": [],
        "regex_patterns": [],
        "min_keyword_hits": 2,
        "seed_queries": [],
    },
    "rag": {
        "top_k_bm25": 20,
        "top_k_dense": 20,
        "top_k_reranked": 5,
        "use_hyde": True,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "guardrails": {
        "min_confidence": 0.55,
        "require_citations": True,
        "hallucination_threshold": 0.35,
        "domain_validation_rules": [],
        "forbidden_phrases": [
            "I think it might be",
            "probably fine",
            "don't worry about it",
        ],
    },
    "knowledge_dir": "knowledge",
    "decision_tree_path": "decision_tree.json",
    "sources_of_truth": [],
    "fallback_message": "I cannot verify this with sufficient confidence. Please consult a certified {domain} expert.",
}


DEFAULT_DECISION_TREE = {
    "name": "{domain} Analysis Decision Tree",
    "version": "1.0.0",
    "root": "start",
    "nodes": [
        {"id": "start", "label": "Identify Query Type", "question": "What type of {domain} question is this?"},
        {"id": "gather_info", "label": "Gather Information", "question": "What specific information is needed?"},
        {"id": "analyze", "label": "Analyze with Domain Knowledge", "question": "Apply domain expertise to analyze the problem."},
        {"id": "validate", "label": "Validate Against Standards", "question": "Check against known standards and best practices."},
        {"id": "respond", "label": "Provide Expert Response", "question": ""},
    ],
    "edges": [
        {"from": "start", "to": "gather_info", "condition": "Query classified"},
        {"from": "gather_info", "to": "analyze", "condition": "Information gathered"},
        {"from": "analyze", "to": "validate", "condition": "Analysis complete"},
        {"from": "validate", "to": "respond", "condition": "Validation passed"},
    ],
}


def create_capsule(domain: str, docs_dir: str | None = None, capsules_base: str = "./capsules"):
    """Create a new capsule directory with template files."""
    import json

    domain_slug = domain.lower().replace(" ", "_").replace("-", "_")
    capsule_dir = Path(capsules_base) / domain_slug
    knowledge_dir = capsule_dir / "knowledge"

    # Create directories
    capsule_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir.mkdir(exist_ok=True)

    # Generate capsule.yaml
    capsule_config = DEFAULT_CAPSULE_TEMPLATE.copy()
    capsule_config["name"] = f"{domain} Expert"
    capsule_config["domain_id"] = domain_slug
    capsule_config["description"] = f"SME capsule for {domain} domain expertise."
    capsule_config["expert_persona"] = capsule_config["expert_persona"].format(domain=domain)
    capsule_config["fallback_message"] = capsule_config["fallback_message"].format(domain=domain)

    yaml_path = capsule_dir / "capsule.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(capsule_config, f, default_flow_style=False, sort_keys=False)

    # Generate decision_tree.json
    dt = DEFAULT_DECISION_TREE.copy()
    dt["name"] = dt["name"].format(domain=domain)
    for node in dt["nodes"]:
        node["question"] = node["question"].format(domain=domain)

    dt_path = capsule_dir / "decision_tree.json"
    with open(dt_path, "w") as f:
        json.dump(dt, f, indent=2)

    console.print(Panel(
        f"[bold green]✅ Capsule '{domain}' created![/]\n\n"
        f"📁 Location: {capsule_dir}\n"
        f"📄 Config: {yaml_path}\n"
        f"🌳 Decision Tree: {dt_path}\n"
        f"📚 Knowledge Dir: {knowledge_dir}\n\n"
        f"[dim]Next steps:[/]\n"
        f"  1. Add domain documents to: {knowledge_dir}/\n"
        f"  2. Edit capsule.yaml to add keywords and seed queries\n"
        f"  3. Customize the decision tree in decision_tree.json\n"
        f"  4. Run: python main.py ingest --domain {domain_slug}",
        title="🧬 New DNA Capsule Created",
        border_style="green",
    ))

    # If docs directory provided, copy files to knowledge dir
    if docs_dir:
        docs_path = Path(docs_dir)
        if docs_path.exists():
            import shutil
            file_count = 0
            for f in docs_path.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".csv"}:
                    shutil.copy2(f, knowledge_dir / f.name)
                    file_count += 1
            console.print(f"[cyan]📂 Copied {file_count} document(s) to knowledge directory.[/]")
        else:
            console.print(f"[yellow]⚠️ Documents directory not found: {docs_dir}[/]")

    return capsule_dir


def ingest_capsule(domain_slug: str, capsules_base: str = "./capsules"):
    """Ingest documents for an existing capsule into the vector store."""
    from core.capsule_loader import CapsuleLoader
    from rag.vector_store import VectorStoreManager
    from rag.ingestion import DocumentIngestionPipeline

    loader = CapsuleLoader(capsules_base)
    capsule = loader.load(domain_slug)

    if not capsule.knowledge_dir:
        console.print(f"[red]❌ No knowledge_dir configured for capsule '{domain_slug}'.[/]")
        return

    vs = VectorStoreManager()
    pipeline = DocumentIngestionPipeline(vs)

    console.print(f"[cyan]📥 Ingesting documents for '{domain_slug}'...[/]")
    summary = pipeline.ingest_capsule(
        domain_id=domain_slug,
        knowledge_dir=capsule.knowledge_dir,
        chunk_size=capsule.rag.chunk_size,
        chunk_overlap=capsule.rag.chunk_overlap,
        embedding_model=capsule.rag.embedding_model,
    )

    console.print(Panel(
        f"[bold green]✅ Ingestion complete![/]\n\n"
        f"📄 Files found: {summary['files_found']}\n"
        f"📦 Chunks created: {summary['chunks_created']}\n"
        f"💾 Chunks stored: {summary['chunks_stored']}",
        title=f"🔍 Ingestion Summary — {domain_slug}",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(description="SME-PLUG Capsule Creator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new DNA capsule")
    create_parser.add_argument("--domain", required=True, help="Domain name (e.g., 'Petroleum Engineering')")
    create_parser.add_argument("--docs", help="Path to documents directory to include")
    create_parser.add_argument("--base-dir", default="./capsules", help="Base capsules directory")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents for existing capsule")
    ingest_parser.add_argument("--domain", required=True, help="Domain slug (e.g., 'petroleum_engineering')")
    ingest_parser.add_argument("--base-dir", default="./capsules", help="Base capsules directory")

    args = parser.parse_args()

    if args.command == "create":
        create_capsule(args.domain, args.docs, args.base_dir)
    elif args.command == "ingest":
        ingest_capsule(args.domain, args.base_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

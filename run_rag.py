# run_rag.py
# -*- coding: utf-8 -*-
"""
RAG pipeline: index specs & analysis, then compare using local LLM.

Usage:
    # Step 1: Index documents
    python run_rag.py index <specs_dir> <analysis_dir>

    # Step 2: Compare (interactive)
    python run_rag.py ask "F_53はどの仕様書に対応していますか？"

    # Step 3: Auto-compare all specs vs analysis
    python run_rag.py compare [-o <output_dir>]
"""

import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_packages():
    missing = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("openai")
    try:
        import chromadb  # noqa: F401
    except ImportError:
        missing.append("chromadb")
    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_packages()

from utils.rag_store import RAGStore  # noqa: E402
from utils.project_utils import load_yaml  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger("RAG", "RAG_")

RAG_DIR = str(SCRIPT_DIR / ".rag_store")


def get_llm() -> LLMClient:
    models = load_yaml(SCRIPT_DIR / "config" / "models.yaml")
    default = models["default_model"]
    cfg = models["models"][default]
    return LLMClient(
        base_url=cfg["api_base"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.2),
        max_tokens=2000,
    )


# ---------------------------------------------------------------------------
# 1. Index
# ---------------------------------------------------------------------------

def cmd_index(specs_dir: str, analysis_dir: str):
    store = RAGStore(persist_dir=RAG_DIR)

    print(f"\n=== Indexing specs: {specs_dir} ===")
    store.index_folder(specs_dir, "specs", extensions=(".md",))

    print(f"\n=== Indexing analysis: {analysis_dir} ===")
    store.index_folder(analysis_dir, "analysis", extensions=(".md", ".json"))

    print("\nIndexing complete.")


# ---------------------------------------------------------------------------
# 2. Ask (interactive query)
# ---------------------------------------------------------------------------

QUERY_SYSTEM = """You are a project analyst for a vehicle meter QML application.
You have access to retrieved context from:
- Design spec documents (Japanese)
- QML source code analysis (functions, signals, properties, bindings)

Answer the user's question based ONLY on the provided context.
If the context is insufficient, say so. Answer in the same language as the question."""


def cmd_ask(question: str):
    store = RAGStore(persist_dir=RAG_DIR)
    llm = get_llm()

    # Retrieve from both collections
    spec_hits = store.query("specs", question, n_results=5)
    analysis_hits = store.query("analysis", question, n_results=5)

    context = "## Spec Documents\n\n"
    for h in spec_hits:
        context += f"[{h['source']}] {h['heading']}\n{h['text']}\n\n"

    context += "## Code Analysis\n\n"
    for h in analysis_hits:
        context += f"[{h['source']}] {h['heading']}\n{h['text']}\n\n"

    messages = [
        {"role": "system", "content": QUERY_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    print(f"\n--- Retrieved {len(spec_hits)} spec + {len(analysis_hits)} analysis chunks ---\n")
    answer = llm.chat(messages, stream=False)
    print(answer)


# ---------------------------------------------------------------------------
# 3. Compare (auto-compare all specs)
# ---------------------------------------------------------------------------

COMPARE_SYSTEM = """You MUST output ONLY a valid JSON object. No explanations, no markdown.

Compare the spec context against the code analysis context.
Output: {"match_rate":"0-100%","implemented":["feature found in both"],"missing":["in spec only"],"extra":["in code only"],"notes":"Japanese observation"}

Rules:
- Match by feature name, function, signal, property, binding, screenId, CAN signal
- Max 10 items per list, each under 50 chars
- notes: one short paragraph in Japanese
- ONLY output JSON"""


def cmd_compare(output_dir: str):
    store = RAGStore(persist_dir=RAG_DIR)
    llm = get_llm()

    # Get all spec sources
    specs_col = store.get_or_create_collection("specs")
    all_specs = specs_col.get(include=["metadatas"])
    spec_sources = sorted(set(m["source"] for m in all_specs["metadatas"]))

    # Filter to main spec files only (skip changelogs)
    spec_sources = [s for s in spec_sources if not s.startswith("変更履歴")]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for spec_source in spec_sources:
        spec_name = Path(spec_source).stem
        logger.info(f"[COMPARE] {spec_name}")

        # Use spec filename + content as query to find relevant analysis
        query = f"{spec_name}"

        spec_hits = store.query("specs", query, n_results=8)
        # Filter to this spec's chunks only
        spec_hits = [h for h in spec_hits if h["source"] == spec_source]
        if not spec_hits:
            # Fallback: query with the source name
            spec_hits = store.query("specs", spec_source, n_results=5)
            spec_hits = [h for h in spec_hits if h["source"] == spec_source]

        analysis_hits = store.query("analysis", query, n_results=8)

        if not spec_hits:
            logger.warning(f"[SKIP] No spec chunks for {spec_name}")
            continue

        # Build context
        spec_context = "\n".join(f"{h['text']}" for h in spec_hits[:5])
        analysis_context = "\n".join(f"[{h['source']}] {h['text']}" for h in analysis_hits[:5])

        user_msg = f"## Spec\n{spec_context}\n\n## Analysis\n{analysis_context}"

        messages = [
            {"role": "system", "content": COMPARE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = llm.chat(messages, stream=False)
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "match_rate": "unknown",
                "implemented": [],
                "missing": [],
                "extra": [],
                "notes": raw.strip()[:200] if raw else "parse error",
            }
        except Exception as e:
            logger.warning(f"[ERROR] {spec_name}: {e}")
            continue

        # Save per-spec
        json_file = out_path / f"{spec_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        md_file = out_path / f"{spec_name}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(_render_md(spec_name, result))

        logger.info(f"[SAVE] {spec_name} → {result.get('match_rate', '?')}")
        summary_rows.append({"spec": spec_name, **result})

    # Index
    index_file = out_path / "index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(_render_index(summary_rows))
    logger.info(f"[SAVE] Index → {index_file}")


def _flatten(val):
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return [str(val)] if val else []


def _render_md(spec_name: str, result: dict) -> str:
    lines = [
        f"# {spec_name}",
        f"",
        f"- **Match Rate**: {result.get('match_rate', 'unknown')}",
        f"",
    ]
    for section, key in [("Implemented", "implemented"),
                         ("Missing", "missing"),
                         ("Extra", "extra")]:
        items = _flatten(result.get(key, []))
        if items:
            lines.append(f"## {section}")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    notes = result.get("notes", "")
    if notes:
        lines.append("## Notes")
        lines.append(str(notes))
        lines.append("")

    return "\n".join(lines)


def _render_index(rows: list[dict]) -> str:
    lines = [
        "# RAG Comparison: Spec vs Implementation",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total**: {len(rows)} spec(s)",
        "",
        "| Spec | Match Rate | Implemented | Missing | Extra |",
        "|------|-----------|-------------|---------|-------|",
    ]
    for r in rows:
        name = r["spec"]
        link = f"[{name}]({name}.md)"
        rate = r.get("match_rate", "?")
        impl = len(_flatten(r.get("implemented", [])))
        miss = len(_flatten(r.get("missing", [])))
        extra = len(_flatten(r.get("extra", [])))
        lines.append(f"| {link} | {rate} | {impl} | {miss} | {extra} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RAG pipeline for spec vs code comparison")
    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index specs and analysis into vector store")
    p_index.add_argument("specs_dir", help="Path to specs folder")
    p_index.add_argument("analysis_dir", help="Path to analysis folder")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question interactively")
    p_ask.add_argument("question", help="Question to ask")

    # compare
    p_compare = sub.add_parser("compare", help="Auto-compare all specs vs analysis")
    p_compare.add_argument("-o", "--output", default="rag_compare",
                           help="Output directory (default: rag_compare/)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.specs_dir, args.analysis_dir)
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "compare":
        cmd_compare(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

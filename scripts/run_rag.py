# run_rag.py
# -*- coding: utf-8 -*-
"""
RAG pipeline: index specs & analysis, then compare using local LLM.

Usage:
    # Step 1: Index documents
    python run_rag.py index <specs_dir> <analysis_dir>

    # Step 2: Compare (interactive)
    python run_rag.py ask "A is implemented according to spec X?"

    # Step 3: Auto-compare all specs vs analysis
    python run_rag.py compare [-o <output_dir>]
"""

import sys
import subprocess
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(SCRIPT_DIR))


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

def cmd_index(specs_dir: str, analysis_dir: str, qml_dir: str = None):
    store = RAGStore(persist_dir=RAG_DIR)

    print(f"\n=== Indexing specs: {specs_dir} ===")
    store.index_folder(specs_dir, "specs", extensions=(".md",), chunk_size=1200)

    print(f"\n=== Indexing analysis: {analysis_dir} ===")
    store.index_folder(analysis_dir, "analysis", extensions=(".md", ".json"))

    if qml_dir:
        print(f"\n=== Indexing QML source: {qml_dir} ===")
        store.index_folder(qml_dir, "qml_source", extensions=(".qml", ".js", ".mjs"), chunk_size=600)

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

COMPARE_SYSTEM = """You MUST output ONLY a valid JSON object. No explanations, no markdown, no code blocks.

You are analyzing an automotive instrument cluster spec (Japanese) against QML source code.

Output exactly this JSON:
{"match_rate":"0-100%","implemented":["..."],"missing":["..."],"extra":["..."],"notes":"..."}

Rules:
- implemented: requirements found in BOTH spec and code (CAN signal handling, display states, property bindings)
- missing: requirements clearly in spec but NOT in code (CAN signal IDs, display conditions, state machine logic)
- extra: code behavior not described in spec
- Match by: CAN signal name (e.g. ACC_CC_SLD_disp2), signal ID ($46C), function name, property name, display state
- Max 10 items per list, each under 60 chars
- notes: one short paragraph in Japanese describing overall gap
- ONLY output the JSON object"""


def _extract_spec_sections(spec_path: Path) -> list[dict]:
    """Split spec MD into H2/H3 sections. Returns list of {heading, content}."""
    text = spec_path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r'(^#{2,3}\s+.+$)', text, flags=re.MULTILINE)

    sections = []
    current_heading = Path(spec_path).stem
    current_content = ""

    for part in parts:
        if re.match(r'^#{2,3}\s+', part):
            if current_content.strip():
                sections.append({"heading": current_heading, "content": current_content.strip()})
            current_heading = part.strip()
            current_content = part + "\n"
        else:
            current_content += part

    if current_content.strip():
        sections.append({"heading": current_heading, "content": current_content.strip()})

    return sections


def _query_code(store: RAGStore, query: str, spec_name: str) -> str:
    """Retrieve relevant code from qml_source (preferred) or analysis collection."""
    try:
        hits = store.query("qml_source", query, n_results=5)
        if hits:
            return "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    except Exception:
        pass

    # Fallback to analysis collection
    hits = store.query("analysis", query, n_results=5)
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)


def cmd_compare(output_dir: str, specs_dir: str = None):
    store = RAGStore(persist_dir=RAG_DIR)
    llm = get_llm()

    # Determine spec files to process
    if specs_dir:
        spec_files = sorted(Path(specs_dir).rglob("*.md"))
    else:
        # Get spec sources from vector store
        specs_col = store.get_or_create_collection("specs")
        all_specs = specs_col.get(include=["metadatas"])
        spec_sources = sorted(set(m["source"] for m in all_specs["metadatas"]))
        config = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
        exclude = config.get("spec", {}).get("exclude_prefixes", [])
        spec_sources = [s for s in spec_sources if not any(s.startswith(p) for p in exclude)]
        spec_files = [SCRIPT_DIR / "specs" / s for s in spec_sources]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for spec_file in spec_files:
        if not spec_file.exists():
            logger.warning(f"[SKIP] Not found: {spec_file}")
            continue

        spec_name = spec_file.stem
        logger.info(f"[COMPARE] {spec_name}")

        # Split spec into sections, focus on 基本動作 (behavior) sections
        sections = _extract_spec_sections(spec_file)
        cfg = load_yaml(SCRIPT_DIR / "config" / "config.yaml").get("spec", {})
        behavior_kw = cfg.get("behavior_headings", [])
        skip_kw = cfg.get("skip_headings", [])
        behavior_sections = [s for s in sections if any(kw in s["heading"] for kw in behavior_kw)]
        if not behavior_sections:
            # Fall back to all sections except input/signal table sections
            behavior_sections = [s for s in sections
                                 if not any(kw in s["heading"] for kw in skip_kw)][:5]
        if not behavior_sections:
            behavior_sections = sections[:5]

        # Per-section gap analysis
        section_results = []
        for sec in behavior_sections:
            query = f"{spec_name} {sec['heading']}"
            code_context = _query_code(store, query, spec_name)

            # Keep section content within ~2000 chars for llama3.1:8b context
            spec_text = sec["content"][:2000]
            code_text = code_context[:2500]

            user_msg = f"## Spec Section: {sec['heading']}\n{spec_text}\n\n## Code\n{code_text}"
            messages = [
                {"role": "system", "content": COMPARE_SYSTEM},
                {"role": "user", "content": user_msg},
            ]

            try:
                raw = llm.chat(messages, stream=False)
                r = json.loads(raw)
                section_results.append(r)
                logger.info(f"  [{sec['heading'][:40]}] → {r.get('match_rate', '?')}")
            except json.JSONDecodeError:
                section_results.append({
                    "match_rate": "unknown",
                    "implemented": [],
                    "missing": [],
                    "extra": [],
                    "notes": raw.strip()[:200] if raw else "parse error",
                })
            except Exception as e:
                logger.warning(f"[ERROR] {spec_name}/{sec['heading']}: {e}")

        # Aggregate section results
        result = _aggregate_results(section_results)

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


def _aggregate_results(results: list[dict]) -> dict:
    """Merge multiple section-level results into one spec-level result."""
    if not results:
        return {"match_rate": "unknown", "implemented": [], "missing": [], "extra": [], "notes": ""}

    implemented = []
    missing = []
    extra = []
    notes_parts = []
    rates = []

    for r in results:
        implemented.extend(r.get("implemented", []))
        missing.extend(r.get("missing", []))
        extra.extend(r.get("extra", []))
        if r.get("notes"):
            notes_parts.append(r["notes"])
        rate_str = r.get("match_rate", "")
        try:
            rates.append(int(str(rate_str).replace("%", "")))
        except (ValueError, TypeError):
            pass

    avg_rate = f"{round(sum(rates) / len(rates))}%" if rates else "unknown"

    # Deduplicate and limit
    def dedup(lst):
        seen = set()
        out = []
        for item in lst:
            key = str(item).lower()[:40]
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out[:15]

    return {
        "match_rate": avg_rate,
        "implemented": dedup(implemented),
        "missing": dedup(missing),
        "extra": dedup(extra),
        "notes": " / ".join(notes_parts)[:500],
    }


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
    p_index.add_argument("--qml", default=None, dest="qml_dir",
                         help="Path to QML source directory (indexes actual source code)")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question interactively")
    p_ask.add_argument("question", help="Question to ask")

    # compare
    p_compare = sub.add_parser("compare", help="Auto-compare all specs vs analysis")
    p_compare.add_argument("-o", "--output", default="rag_compare",
                           help="Output directory (default: rag_compare/)")
    p_compare.add_argument("--specs", default=None,
                           help="Specs directory to read directly (bypasses vector store for spec side)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.specs_dir, args.analysis_dir, qml_dir=args.qml_dir)
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "compare":
        cmd_compare(args.output, specs_dir=args.specs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

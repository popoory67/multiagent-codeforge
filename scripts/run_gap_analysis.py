# scripts/run_gap_analysis.py
# -*- coding: utf-8 -*-
"""
End-to-end gap analysis orchestrator.

Runs the full pipeline in order:
  1. Check Ollama is running
  2. Generate prompts (if missing)
  3. Analyze QML source code
  4. Index specs + analysis + QML into RAG store
  5. Run gap analysis (spec vs QML)
  6. Print summary

Usage:
    # Use qml_root from config.yaml
    python scripts/run_gap_analysis.py

    # Override qml_root on the command line
    python scripts/run_gap_analysis.py --qml /path/to/qml/src

    # Skip steps already completed
    python scripts/run_gap_analysis.py --skip-analyze --skip-index

    # Output to custom directory
    python scripts/run_gap_analysis.py -o my_report/
"""

import sys
import subprocess
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(SCRIPT_DIR))


def _ensure_packages():
    missing = []
    for pkg in ["yaml", "openai", "chromadb", "requests"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append({"yaml": "pyyaml"}.get(pkg, pkg))
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_packages()

from utils.project_utils import load_yaml          # noqa: E402
from utils.llm_client import assert_ollama         # noqa: E402
from utils.logger import setup_logger              # noqa: E402
from utils.rag_store import RAGStore               # noqa: E402

logger = setup_logger("GapAnalysis", "GapAnalysis_")

SPECS_DIR  = SCRIPT_DIR / "specs"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
RAG_DIR    = str(SCRIPT_DIR / ".rag_store")


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def step(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}")


def run_script(script: str, *args: str):
    """Run a script in the same Python interpreter and return exit code."""
    cmd = [sys.executable, str(SCRIPT_DIR / "scripts" / script), *args]
    logger.info(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    return result.returncode


# ---------------------------------------------------------------------------
# Step 1: Ollama health check
# ---------------------------------------------------------------------------

def step_check_ollama(api_base: str):
    step(1, "Ollama health check")
    try:
        assert_ollama(api_base)
        print(f"  OK  Ollama is running at {api_base.replace('/v1','')}")
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2: Generate prompts (if missing)
# ---------------------------------------------------------------------------

def step_generate_prompts():
    step(2, "Generate prompts (llama)")
    compare_md = PROMPTS_DIR / "compare.md"
    analyze_md  = PROMPTS_DIR / "analyze.md"

    missing = []
    if not compare_md.exists() or compare_md.stat().st_size < 100:
        missing.append("compare")
    if not analyze_md.exists() or analyze_md.stat().st_size < 100:
        missing.append("analyze")

    if not missing:
        print("  OK  Prompts already exist — skipping")
        print(f"       compare: {compare_md}")
        print(f"       analyze: {analyze_md}")
        print("  TIP: Delete prompt files and re-run to regenerate them.")
        return

    for t in missing:
        print(f"  Generating prompts/{t}.md via llama ...")
        rc = run_script("run_generate_prompts.py", "--type", t, "--specs", str(SPECS_DIR))
        if rc != 0:
            print(f"  WARNING: prompt generation failed for '{t}' (exit {rc}). Using defaults.")


# ---------------------------------------------------------------------------
# Step 3: Analyze QML source
# ---------------------------------------------------------------------------

def step_analyze(qml_root: str, skip: bool) -> Path:
    step(3, f"Analyze QML source: {qml_root}")

    if skip:
        # Find the latest existing analysis
        analysis_root = SCRIPT_DIR / "analysis"
        dirs = sorted(analysis_root.iterdir()) if analysis_root.exists() else []
        dirs = [d for d in dirs if d.is_dir()]
        if dirs:
            latest = dirs[-1]
            print(f"  SKIP  Using existing analysis: {latest}")
            return latest
        else:
            print("  WARNING: --skip-analyze given but no analysis found. Running analyze...")

    if not qml_root or not Path(qml_root).is_dir():
        print(f"\n  ERROR: qml_root is not a valid directory: '{qml_root}'")
        print("  Set 'project.qml_root' in config/config.yaml or pass --qml <path>")
        sys.exit(1)

    rc = run_script("run_analyze.py", qml_root, "--no-llm")
    if rc != 0:
        print(f"  ERROR: run_analyze.py exited with code {rc}")
        sys.exit(1)

    # Find the newly created analysis dir (latest timestamp)
    analysis_root = SCRIPT_DIR / "analysis"
    dirs = sorted(d for d in analysis_root.iterdir() if d.is_dir())
    if not dirs:
        print("  ERROR: No analysis output found after running run_analyze.py")
        sys.exit(1)

    latest = dirs[-1]
    print(f"  OK  Analysis saved to: {latest}")
    return latest


# ---------------------------------------------------------------------------
# Step 4: Index into RAG store
# ---------------------------------------------------------------------------

def step_index(analysis_dir: Path, qml_root: str, skip: bool):
    step(4, "Index specs + analysis + QML into RAG store")

    if skip:
        print("  SKIP  --skip-index given")
        return

    store = RAGStore(persist_dir=RAG_DIR)

    print(f"  Indexing specs: {SPECS_DIR}")
    store.index_folder(str(SPECS_DIR), "specs", extensions=(".md",), chunk_size=1200)

    print(f"  Indexing analysis: {analysis_dir}")
    store.index_folder(str(analysis_dir), "analysis", extensions=(".md", ".json"))

    if qml_root and Path(qml_root).is_dir():
        print(f"  Indexing QML source: {qml_root}")
        store.index_folder(qml_root, "qml_source", extensions=(".qml", ".js", ".mjs"), chunk_size=600)
    else:
        print("  NOTE: qml_root not set — skipping QML source indexing")
        print("        Set 'project.qml_root' in config/config.yaml for better results")

    print("  OK  Indexing complete")


# ---------------------------------------------------------------------------
# Step 5: Gap analysis
# ---------------------------------------------------------------------------

def step_gap_analysis(output_dir: Path, skip: bool):
    step(5, f"Gap analysis → {output_dir}")

    if skip:
        print("  SKIP  --skip-compare given")
        return

    rc = run_script(
        "run_rag.py", "compare",
        "--specs", str(SPECS_DIR),
        "-o", str(output_dir),
    )
    if rc != 0:
        print(f"  ERROR: run_rag.py compare exited with code {rc}")
        sys.exit(1)

    print(f"  OK  Gap analysis saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Step 6: Summary
# ---------------------------------------------------------------------------

def step_summary(output_dir: Path):
    step(6, "Summary")
    index_file = output_dir / "index.md"
    if not index_file.exists():
        print("  No index.md found.")
        return

    lines = index_file.read_text(encoding="utf-8").splitlines()
    # Print the markdown table rows
    in_table = False
    total = impl = miss = 0
    for line in lines:
        if line.startswith("|"):
            in_table = True
            print(f"  {line}")
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 4 and cols[1] not in ("Match Rate", "---"):
                try:
                    impl += int(cols[2])
                    miss += int(cols[3])
                    total += 1
                except ValueError:
                    pass
        elif in_table and not line.startswith("|"):
            break

    print(f"\n  Specs analyzed : {total}")
    print(f"  Implemented    : {impl} items")
    print(f"  Missing        : {miss} items")
    print(f"\n  Full report: {output_dir / 'index.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="End-to-end gap analysis: spec requirements vs QML source code"
    )
    parser.add_argument("--qml", default=None,
                        help="Path to QML source root (overrides config.yaml qml_root)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: gap_report/YYYYMMDD_HHMMSS/)")
    parser.add_argument("--skip-prompts",  action="store_true", help="Skip prompt generation")
    parser.add_argument("--skip-analyze",  action="store_true", help="Skip QML analysis (use latest)")
    parser.add_argument("--skip-index",    action="store_true", help="Skip RAG indexing")
    parser.add_argument("--skip-compare",  action="store_true", help="Skip gap analysis")
    args = parser.parse_args()

    # Load config
    config = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
    models = load_yaml(SCRIPT_DIR / "config" / "models.yaml")
    default = models["default_model"]
    api_base = models["models"][default]["api_base"]

    qml_root = args.qml or config.get("project", {}).get("qml_root", "")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else SCRIPT_DIR / "gap_report" / timestamp

    print(f"\n{'#'*60}")
    print(f"  GAP ANALYSIS PIPELINE")
    print(f"  QML root : {qml_root or '(not set)'}")
    print(f"  Specs    : {SPECS_DIR}")
    print(f"  Output   : {output_dir}")
    print(f"{'#'*60}")

    step_check_ollama(api_base)

    if not args.skip_prompts:
        step_generate_prompts()
    else:
        print("\n  STEP 2: SKIPPED (--skip-prompts)")

    analysis_dir = step_analyze(qml_root, skip=args.skip_analyze)
    step_index(analysis_dir, qml_root, skip=args.skip_index)
    step_gap_analysis(output_dir, skip=args.skip_compare)
    step_summary(output_dir)


if __name__ == "__main__":
    main()

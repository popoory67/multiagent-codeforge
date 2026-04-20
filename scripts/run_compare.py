# run_compare.py
# -*- coding: utf-8 -*-
"""
Compare spec documents against QML analysis results using local LLM.

Usage:
    python run_compare.py <analysis_dir> [-o <output_dir>]

    analysis_dir: output from run_analyze.py (e.g. analysis/2026-04-02_12-00-00)
    -o: output directory for comparison results (default: compare/)

Output:
    compare/
        index.md          # summary table of all specs
        <spec_name>.md    # detailed comparison for each spec
        <spec_name>.json  # raw JSON results from LLM for each spec
    
Reports:
    match_rate (0~100%)
    implemented — The features that are correctly implemented according to the spec
    missing — The features that are in the spec but not implemented
    extra — The features that are implemented but not in the spec (potential over-implementation)
    notes — The features that are in the spec but not implemented
"""

import sys
import subprocess
import json
import re
import asyncio
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
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_packages()

from utils.project_utils import load_yaml  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402
from utils.async_executor import run_async_jobs  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger("Compare", "Compare_")


# ---------------------------------------------------------------------------
# 1. Load inputs
# ---------------------------------------------------------------------------

def load_compare_map() -> list[dict]:
    path = SCRIPT_DIR / "config" / "compare_map.yaml" # TODO: temporal yaml file
    data = load_yaml(path)
    return data.get("mappings", [])


def load_prompt() -> str:
    path = SCRIPT_DIR / "prompts" / "compare.md"
    return path.read_text(encoding="utf-8")


def load_spec(spec_path: str) -> str:
    p = SCRIPT_DIR / spec_path
    if not p.exists():
        raise FileNotFoundError(f"Spec not found: {p}")
    return p.read_text(encoding="utf-8")


def load_analysis_modules(analysis_dir: Path, module_names: list[str]) -> str:
    """Load and concatenate analysis MD files for the given module names."""
    parts = []
    for name in module_names:
        # Try both the module dir structure and flat structure
        candidates = [
            analysis_dir / name / f"{name}.md",
            analysis_dir / name / f"{name}.json",
            analysis_dir / f"{name}.md",
            analysis_dir / f"{name}.json",
        ]
        for c in candidates:
            if c.exists():
                parts.append(f"--- {c.name} ---\n{c.read_text(encoding='utf-8')}")
                break
        else:
            parts.append(f"--- {name} ---\n(not found)")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2. LLM comparison job
# ---------------------------------------------------------------------------

def _split_spec_sections(text: str) -> list[dict]:
    """Split spec MD into H2/H3 sections. Returns [{heading, content}]."""
    parts = re.split(r'(^#{2,3}\s+.+$)', text, flags=re.MULTILINE)
    sections = []
    current_heading = ""
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


def _aggregate_section_results(results: list[dict]) -> dict:
    """Merge multiple section-level LLM results into one spec-level result."""
    if not results:
        return {"match_rate": "unknown", "implemented": [], "missing": [], "extra": [], "notes": ""}
    implemented, missing, extra, notes_parts, rates = [], [], [], [], []
    for r in results:
        implemented.extend(r.get("implemented", []))
        missing.extend(r.get("missing", []))
        extra.extend(r.get("extra", []))
        if r.get("notes"):
            notes_parts.append(r["notes"])
        try:
            rates.append(int(str(r.get("match_rate", "0")).replace("%", "")))
        except (ValueError, TypeError):
            pass

    def dedup(lst):
        seen, out = set(), []
        for item in lst:
            k = str(item).lower()[:40]
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out[:15]

    avg_rate = f"{round(sum(rates) / len(rates))}%" if rates else "unknown"
    return {
        "match_rate": avg_rate,
        "implemented": dedup(implemented),
        "missing": dedup(missing),
        "extra": dedup(extra),
        "notes": " / ".join(notes_parts)[:500],
    }


class CompareJob:
    def __init__(self, job_id: int, spec_name: str, spec_text: str,
                 analysis_text: str, prompt: str, llm: LLMClient):
        self.id = job_id
        self.spec_name = spec_name
        self.spec_text = spec_text
        self.analysis_text = analysis_text
        self.prompt = prompt
        self.llm = llm

    def _run_section(self, heading: str, spec_content: str) -> dict:
        """Run LLM comparison for a single spec section."""
        # llama3.1:8b: ~8k token context → keep inputs under ~2500 chars each
        spec = spec_content[:2500]
        analysis = self.analysis_text[:2500]
        user_msg = (
            f"## Spec Section: {heading}\n\n{spec}\n\n"
            f"## Code Analysis\n\n{analysis}"
        )
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.chat(messages, stream=False)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"match_rate": "unknown", "implemented": [], "missing": [],
                    "extra": [], "notes": raw.strip()[:200]}

    def run(self):
        # Split spec into sections; focus on 基本動作 (behavior) sections
        sections = _split_spec_sections(self.spec_text)
        behavior = [s for s in sections if "基本動作" in s["heading"] or "動作" in s["heading"]]
        if not behavior:
            # Fall back: skip 入力 (huge CAN tables), use remaining sections
            behavior = [s for s in sections if "入力" not in s["heading"]][:5]
        if not behavior:
            behavior = sections[:5]

        section_results = []
        for sec in behavior:
            r = self._run_section(sec["heading"], sec["content"])
            section_results.append(r)

        result = _aggregate_section_results(section_results)
        return {"id": self.id, "spec": self.spec_name, "result": result, "error": None}


# ---------------------------------------------------------------------------
# 3. Orchestrator
# ---------------------------------------------------------------------------

async def compare(analysis_dir: str, output_dir: str):
    analysis_path = Path(analysis_dir)
    if not analysis_path.is_dir():
        logger.error(f"Analysis directory not found: {analysis_dir}")
        sys.exit(1)

    # Load config
    mappings = load_compare_map()
    prompt = load_prompt()

    models = load_yaml(SCRIPT_DIR / "config" / "models.yaml")
    default = models["default_model"]
    model_cfg = models["models"][default]

    llm = LLMClient(
        base_url=model_cfg["api_base"],
        api_key=model_cfg["api_key"],
        model=model_cfg["model"],
        temperature=model_cfg.get("temperature", 0.2),
    )

    # Build jobs
    jobs = []
    for i, mapping in enumerate(mappings):
        spec_path = mapping["spec"]
        module_names = mapping["modules"]

        try:
            spec_text = load_spec(spec_path)
        except FileNotFoundError as e:
            logger.warning(f"[SKIP] {e}")
            continue

        analysis_text = load_analysis_modules(analysis_path, module_names)
        spec_name = Path(spec_path).stem

        jobs.append(CompareJob(i, spec_name, spec_text, analysis_text, prompt, llm))
        logger.info(f"[JOB] {spec_name} ↔ {module_names}")

    if not jobs:
        logger.error("No valid spec-module pairs found.")
        sys.exit(1)

    logger.info(f"[RUN] Comparing {len(jobs)} spec(s) via LLM...")
    results = await run_async_jobs(jobs, workers=2)

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in results:
        if r.get("error"):
            logger.warning(f"[ERROR] {r.get('spec', '?')}: {r['error']}")
            continue

        spec_name = r["spec"]
        result = r["result"]

        # Per-spec JSON
        json_file = out_path / f"{spec_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Per-spec MD
        md_file = out_path / f"{spec_name}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(render_compare_md(spec_name, result))

        logger.info(f"[SAVE] {spec_name} → {out_path}")
        summary_rows.append({"spec": spec_name, **result})

    # Index
    index_file = out_path / "index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(render_index(summary_rows))
    logger.info(f"[SAVE] Index → {index_file}")


# ---------------------------------------------------------------------------
# 4. Renderers
# ---------------------------------------------------------------------------

def render_compare_md(spec_name: str, result: dict) -> str:
    lines = [
        f"# Comparison: {spec_name}",
        f"",
        f"- **Match Rate**: {result.get('match_rate', 'unknown')}",
        f"",
    ]

    def _flatten(val):
        if isinstance(val, list):
            return [str(v) for v in val]
        if isinstance(val, str):
            return [val] if val else []
        return [str(val)] if val else []

    impl = _flatten(result.get("implemented", []))
    if impl:
        lines.append("## Implemented")
        for item in impl:
            lines.append(f"- {item}")
        lines.append("")

    missing = _flatten(result.get("missing", []))
    if missing:
        lines.append("## Missing")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")

    extra = _flatten(result.get("extra", []))
    if extra:
        lines.append("## Extra (not in spec)")
        for item in extra:
            lines.append(f"- {item}")
        lines.append("")

    notes = result.get("notes", "")
    if notes:
        lines.append("## Notes")
        lines.append(str(notes))
        lines.append("")

    return "\n".join(lines)


def render_index(rows: list[dict]) -> str:
    lines = [
        f"# Spec vs Implementation Comparison",
        f"",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total**: {len(rows)} spec(s)",
        f"",
        "| Spec | Match Rate | Implemented | Missing | Extra |",
        "|------|-----------|-------------|---------|-------|",
    ]
    for r in rows:
        name = r["spec"]
        link = f"[{name}]({name}.md)"
        rate = r.get("match_rate", "?")
        impl = len(r.get("implemented", []))
        miss = len(r.get("missing", []))
        extra = len(r.get("extra", []))
        lines.append(f"| {link} | {rate} | {impl} | {miss} | {extra} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser(description="Compare specs against QML analysis")
    parser.add_argument("analysis_dir", help="Path to analysis output directory")
    parser.add_argument("-o", "--output", default="compare",
                        help="Output directory (default: compare/)")
    args = parser.parse_args()

    asyncio.run(compare(args.analysis_dir, args.output))


if __name__ == "__main__":
    main()

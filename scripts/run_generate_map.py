# run_generate_map.py
# -*- coding: utf-8 -*-
"""
Auto-generate compare_map.yaml by matching spec MDs with analysis module MDs.

Usage:
    python run_generate_map.py <specs_dir> <analysis_dir> [-o <output_yaml>] [--llm]

    Default: keyword matching (fast, no LLM)
    --llm:   use local Ollama LLM to refine matching
"""

import os
import sys
import subprocess
import re
import json
import asyncio
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(SCRIPT_DIR))


def _ensure_packages(use_llm: bool = False):
    missing = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    if use_llm:
        try:
            import openai  # noqa: F401
        except ImportError:
            missing.append("openai")
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_use_llm = "--llm" in sys.argv
_ensure_packages(use_llm=_use_llm)

import yaml  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

if _use_llm:
    from utils.project_utils import load_yaml  # noqa: E402
    from utils.llm_client import LLMClient  # noqa: E402
    from utils.async_executor import run_async_jobs  # noqa: E402

logger = setup_logger("GenMap", "GenMap_")


# ---------------------------------------------------------------------------
# 1. Extract keywords from files
# ---------------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 50) -> set[str]:
    """Extract meaningful keywords from text (handles Japanese + English)."""
    # English words (3+ chars)
    en_words = set(re.findall(r'[a-zA-Z_]\w{2,}', text.lower()))

    # Japanese keywords: katakana sequences (2+ chars)
    katakana = set(re.findall(r'[\u30A0-\u30FF]{2,}', text))

    # Japanese keywords: kanji sequences (2+ chars)
    kanji = set(re.findall(r'[\u4E00-\u9FFF]{2,}', text))

    all_kw = en_words | katakana | kanji

    # Remove very common/noise words
    noise = {'function', 'property', 'signal', 'import', 'true', 'false',
             'string', 'int', 'bool', 'var', 'void', 'return', 'null',
             'undefined', 'the', 'and', 'for', 'this', 'that', 'with'}
    return all_kw - noise


def load_file_keywords(filepath: Path) -> tuple[str, set[str]]:
    """Return (filename_stem, keywords) for a file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    keywords = extract_keywords(text)
    # Also add filename parts as keywords
    name_parts = set(re.findall(r'[a-zA-Z_]\w{2,}', filepath.stem.lower()))
    katakana_parts = set(re.findall(r'[\u30A0-\u30FF]{2,}', filepath.stem))
    kanji_parts = set(re.findall(r'[\u4E00-\u9FFF]{2,}', filepath.stem))
    keywords |= name_parts | katakana_parts | kanji_parts
    return filepath.stem, keywords


# ---------------------------------------------------------------------------
# 2. Keyword-based matching
# ---------------------------------------------------------------------------

def keyword_match(specs: dict[str, set[str]],
                  modules: dict[str, set[str]],
                  threshold: float = 0.05) -> list[dict]:
    """Match specs to modules by keyword overlap (Jaccard-like score)."""
    results = []

    for spec_path, spec_kw in specs.items():
        scores = []
        for mod_name, mod_kw in modules.items():
            if not spec_kw or not mod_kw:
                continue
            intersection = spec_kw & mod_kw
            union = spec_kw | mod_kw
            score = len(intersection) / len(union) if union else 0
            if score >= threshold:
                scores.append((mod_name, score, intersection))

        scores.sort(key=lambda x: x[1], reverse=True)

        matched_modules = [name for name, _, _ in scores[:5]]  # top 5
        match_detail = {name: {
            "score": round(score, 3),
            "shared_keywords": sorted(list(shared))[:10]
        } for name, score, shared in scores[:5]}

        results.append({
            "spec": spec_path,
            "modules": matched_modules,
            "detail": match_detail,
        })

    return results


# ---------------------------------------------------------------------------
# 3. LLM refinement
# ---------------------------------------------------------------------------

MATCH_SYSTEM = """You are matching design specification documents to QML source code modules.

Given a spec filename and its top keywords, plus a list of candidate modules with their keywords,
select the modules that are most likely implementing the spec.

Output a JSON object:
{
  "modules": ["module_name_1", "module_name_2"],
  "confidence": "high/medium/low",
  "reason": "brief explanation in Japanese"
}

Output valid JSON only. No markdown fences."""


class MatchJob:
    def __init__(self, job_id, spec_path, spec_keywords, candidates, llm):
        self.id = job_id
        self.spec_path = spec_path
        self.spec_keywords = spec_keywords
        self.candidates = candidates
        self.llm = llm

    def run(self):
        user_msg = json.dumps({
            "spec": self.spec_path,
            "spec_keywords": sorted(list(self.spec_keywords))[:30],
            "candidates": {
                name: sorted(list(kw))[:20]
                for name, kw in self.candidates.items()
            }
        }, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": MATCH_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.chat(messages, stream=False)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"modules": [], "confidence": "low", "reason": raw.strip()}

        return {"id": self.id, "spec": self.spec_path, "result": result, "error": None}


async def llm_refine(specs: dict[str, set[str]],
                     modules: dict[str, set[str]],
                     keyword_results: list[dict]) -> list[dict]:
    """Use LLM to refine keyword matching results."""
    models_cfg = load_yaml(SCRIPT_DIR / "config" / "models.yaml")
    default = models_cfg["default_model"]
    model_cfg = models_cfg["models"][default]

    llm = LLMClient(
        base_url=model_cfg["api_base"],
        api_key=model_cfg["api_key"],
        model=model_cfg["model"],
        temperature=model_cfg.get("temperature", 0.2),
    )

    jobs = []
    for i, kr in enumerate(keyword_results):
        spec_path = kr["spec"]
        spec_kw = specs[spec_path]

        # Send top keyword candidates + all modules as fallback
        candidate_names = kr["modules"] if kr["modules"] else list(modules.keys())[:10]
        candidates = {name: modules[name] for name in candidate_names if name in modules}

        jobs.append(MatchJob(i, spec_path, spec_kw, candidates, llm))

    logger.info(f"[LLM] Refining {len(jobs)} matches...")
    results = await run_async_jobs(jobs, workers=2)

    refined = []
    for r in results:
        if r.get("error"):
            logger.warning(f"[LLM] {r.get('spec', '?')}: {r['error']}")
            # Fall back to keyword result
            kr = keyword_results[r["id"]]
            refined.append(kr)
            continue

        kr = keyword_results[r["id"]]
        llm_result = r["result"]
        refined.append({
            "spec": kr["spec"],
            "modules": llm_result.get("modules", kr["modules"]),
            "confidence": llm_result.get("confidence", "low"),
            "reason": llm_result.get("reason", ""),
        })

    return refined


# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------

def save_yaml(mappings: list[dict], output_path: Path):
    """Save mappings as compare_map.yaml format."""
    data = {
        "mappings": [
            {"spec": m["spec"], "modules": m["modules"]}
            for m in mappings
            if m["modules"]
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated spec <-> module mapping\n")
        f.write("# Review and adjust as needed\n\n")
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"[SAVE] {output_path} ({len(data['mappings'])} mappings)")


def save_detail_md(mappings: list[dict], output_path: Path):
    """Save detailed matching report as MD."""
    lines = [
        "# Auto-generated Spec ↔ Module Mapping",
        "",
        "| Spec | Modules | Detail |",
        "|------|---------|--------|",
    ]
    for m in mappings:
        spec = Path(m["spec"]).stem
        mods = ", ".join(m["modules"]) if m["modules"] else "(no match)"
        detail = m.get("confidence", "") or ""
        if "reason" in m:
            detail += f" {m['reason']}"
        if "detail" in m:
            top = list(m["detail"].items())[:2]
            detail = "; ".join(
                f"{name}({d['score']}) [{','.join(d['shared_keywords'][:3])}]"
                for name, d in top
            )
        lines.append(f"| {spec} | {mods} | {detail} |")

    md_path = output_path.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[SAVE] {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser(description="Auto-generate compare_map.yaml")
    parser.add_argument("specs_dir", help="Path to specs folder (MD files)")
    parser.add_argument("analysis_dir", help="Path to analysis output folder")
    parser.add_argument("-o", "--output", default="config/compare_map.yaml",
                        help="Output YAML path (default: config/compare_map.yaml)")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM to refine keyword matching")
    args = parser.parse_args()

    specs_dir = Path(args.specs_dir)
    analysis_dir = Path(args.analysis_dir)

    if not specs_dir.is_dir():
        print(f"Specs directory not found: {specs_dir}")
        sys.exit(1)
    if not analysis_dir.is_dir():
        print(f"Analysis directory not found: {analysis_dir}")
        sys.exit(1)

    # Collect spec files (skip changelog files)
    spec_files = sorted(
        f for f in specs_dir.glob("*.md")
        if not f.stem.startswith("変更履歴")
    )
    logger.info(f"[SCAN] {len(spec_files)} spec files")

    # Collect analysis module MDs
    module_dirs = sorted(
        d for d in analysis_dir.iterdir()
        if d.is_dir()
    )
    logger.info(f"[SCAN] {len(module_dirs)} analysis modules")

    # Extract keywords
    specs = {}
    for f in spec_files:
        stem, kw = load_file_keywords(f)
        specs[str(f)] = kw
        logger.info(f"[KW] spec: {f.name} ({len(kw)} keywords)")

    modules = {}
    for d in module_dirs:
        md_files = list(d.glob("*.md")) + list(d.glob("*.json"))
        if not md_files:
            continue
        combined_text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in md_files)
        kw = extract_keywords(combined_text)
        modules[d.name] = kw
        logger.info(f"[KW] module: {d.name} ({len(kw)} keywords)")

    # Keyword matching
    results = keyword_match(specs, modules)
    logger.info(f"[MATCH] Keyword matching done")

    # LLM refinement
    if args.llm:
        results = asyncio.run(llm_refine(specs, modules, results))
        logger.info(f"[MATCH] LLM refinement done")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(results, output_path)
    save_detail_md(results, output_path)


if __name__ == "__main__":
    main()

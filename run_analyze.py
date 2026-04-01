# run_analyze.py
# -*- coding: utf-8 -*-
"""
Analyze a QML project by directory (module), extract structure,
and optionally use LLM to summarize each module's purpose/status.
"""

import os
import sys
import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from utils.qml_parser import parse_qml_file
from utils.project_utils import load_yaml
from utils.llm_client import LLMClient
from utils.async_executor import run_async_jobs
from utils.logger import setup_logger

SCRIPT_DIR = Path(__file__).resolve().parent
logger = setup_logger("Analyze", "Analyze_")

# ---------------------------------------------------------------------------
# 1. Scan & group
# ---------------------------------------------------------------------------

def scan_qml_project(root: str, target_exts: tuple = (".qml",)) -> dict:
    """Walk *root* and group files by directory."""
    modules: dict[str, list[str]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(target_exts):
                full = os.path.join(dirpath, fn)
                rel_dir = os.path.relpath(dirpath, root) or "."
                modules[rel_dir].append(full)
    return dict(modules)


def dir_hash(files: list[str]) -> str:
    """MD5 over sorted file contents — used to skip unchanged modules."""
    h = hashlib.md5()
    for f in sorted(files):
        try:
            h.update(Path(f).read_bytes())
        except OSError:
            pass
    return h.hexdigest()

# ---------------------------------------------------------------------------
# 2. Static parse per module
# ---------------------------------------------------------------------------

def analyze_module_static(module_path: str, files: list[str]) -> dict:
    """Parse all QML files in a module directory (no LLM)."""
    parsed_files = []
    all_functions = []
    all_signals = []
    all_properties = []
    all_children = set()

    for f in sorted(files):
        try:
            info = parse_qml_file(f)
            parsed_files.append(info)
            for func in info["functions"]:
                func["defined_in"] = info["file"]
                all_functions.append(func)
            for sig in info["signals"]:
                sig["defined_in"] = info["file"]
                all_signals.append(sig)
            for prop in info["properties"]:
                prop["defined_in"] = info["file"]
                all_properties.append(prop)
            all_children.update(info["children"])
        except Exception as e:
            logger.warning(f"[PARSE] Failed {f}: {e}")

    return {
        "path": module_path,
        "file_count": len(files),
        "files": parsed_files,
        "functions": all_functions,
        "signals": all_signals,
        "properties": all_properties,
        "child_components": sorted(all_children),
    }

# ---------------------------------------------------------------------------
# 3. LLM summary job (optional)
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You are a QML project analyst. "
    "Given the parsed structure of a QML module, produce a short JSON object with:\n"
    '  "summary": one-line description of the module\'s purpose,\n'
    '  "status": one of "implemented", "partial", "stub", "unknown",\n'
    '  "issues": list of potential problems (empty list if none).\n'
    "Output valid JSON only. No markdown fences."
)


class SummaryJob:
    def __init__(self, job_id, module_data, llm):
        self.id = job_id
        self.module_data = module_data
        self.llm = llm

    def run(self):
        compact = {
            "path": self.module_data["path"],
            "files": [f["file"] for f in self.module_data["files"]],
            "functions": self.module_data["functions"],
            "signals": self.module_data["signals"],
            "properties": self.module_data["properties"],
            "child_components": self.module_data["child_components"],
        }
        user_msg = json.dumps(compact, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.chat(messages, stream=False)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"summary": raw.strip(), "status": "unknown", "issues": []}

        return {"id": self.id, "result": result, "error": None}

# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------

async def analyze(target_root: str, use_llm: bool = True):
    config = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
    target_exts = tuple(config["project"]["target_exts"])

    logger.info(f"[SCAN] Scanning {target_root} ...")
    modules = scan_qml_project(target_root, target_exts)
    logger.info(f"[SCAN] Found {len(modules)} module(s)")

    # Static parse
    module_data = []
    for mod_path, files in sorted(modules.items()):
        data = analyze_module_static(mod_path, files)
        data["hash"] = dir_hash(files)
        module_data.append(data)
        logger.info(f"[STATIC] {mod_path}: {data['file_count']} files, "
                     f"{len(data['functions'])} funcs, {len(data['signals'])} signals")

    # LLM summary (optional)
    if use_llm:
        models = load_yaml(SCRIPT_DIR / "config" / "models.yaml")
        default = models["default_model"]
        model_cfg = models["models"][default]

        llm = LLMClient(
            base_url=model_cfg["api_base"],
            api_key=model_cfg["api_key"],
            model=model_cfg["model"],
            temperature=model_cfg.get("temperature", 0.2),
        )

        jobs = [
            SummaryJob(i, md, llm)
            for i, md in enumerate(module_data)
        ]

        summaries = await run_async_jobs(jobs, workers=4)

        for s in summaries:
            if s.get("error"):
                logger.warning(f"[LLM] Module {s['id']} summary failed: {s['error']}")
                continue
            idx = s["id"]
            module_data[idx]["summary"] = s["result"].get("summary", "")
            module_data[idx]["status"] = s["result"].get("status", "unknown")
            module_data[idx]["issues"] = s["result"].get("issues", [])
    else:
        for md in module_data:
            md["summary"] = ""
            md["status"] = "unknown"
            md["issues"] = []

    # Build final report
    report = {
        "project": os.path.basename(os.path.abspath(target_root)),
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": os.path.abspath(target_root),
        "module_count": len(module_data),
        "modules": module_data,
    }

    # Save JSON
    os.makedirs("analysis", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join("analysis", f"analysis_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"[SAVE] JSON → {json_path}")

    # Save MD
    md_path = os.path.join("analysis", f"analysis_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    logger.info(f"[SAVE] MD   → {md_path}")

    return report

# ---------------------------------------------------------------------------
# 5. Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    lines = [
        f"# Project Analysis: {report['project']}",
        f"",
        f"- **Date**: {report['analyzed_at']}",
        f"- **Root**: `{report['root']}`",
        f"- **Modules**: {report['module_count']}",
        f"",
    ]

    for mod in report["modules"]:
        status = mod.get("status", "unknown")
        lines.append(f"---")
        lines.append(f"## `{mod['path']}`  [{status}]")
        lines.append(f"")

        summary = mod.get("summary", "")
        if summary:
            lines.append(f"> {summary}")
            lines.append(f"")

        lines.append(f"Files ({mod['file_count']}): "
                      + ", ".join(f"`{fi['file']}`" for fi in mod["files"]))
        lines.append(f"")

        if mod["functions"]:
            lines.append(f"### Functions")
            for fn in mod["functions"]:
                params = fn.get("params", "")
                ret = f" : {fn['return_type']}" if fn.get("return_type") else ""
                lines.append(f"- `{fn['name']}({params}){ret}` — {fn.get('defined_in', '')}")
            lines.append(f"")

        if mod["signals"]:
            lines.append(f"### Signals")
            for sig in mod["signals"]:
                p = f"({sig['params']})" if sig.get("params") else ""
                lines.append(f"- `{sig['name']}{p}` — {sig.get('defined_in', '')}")
            lines.append(f"")

        if mod["properties"]:
            lines.append(f"### Properties")
            for prop in mod["properties"]:
                q = f"[{prop['qualifier']}] " if prop.get("qualifier") else ""
                lines.append(f"- {q}`{prop['type']} {prop['name']}` — {prop.get('defined_in', '')}")
            lines.append(f"")

        issues = mod.get("issues", [])
        if issues:
            lines.append(f"### Issues")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append(f"")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analyze.py <project_root> [--no-llm]")
        sys.exit(1)

    target = sys.argv[1]
    use_llm = "--no-llm" not in sys.argv

    if not os.path.isdir(target):
        print(f"Error: {target} is not a directory")
        sys.exit(1)

    asyncio.run(analyze(target, use_llm=use_llm))


if __name__ == "__main__":
    main()

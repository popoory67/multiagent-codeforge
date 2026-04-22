# scripts/run_generate_prompts.py
# -*- coding: utf-8 -*-
"""
Use Ollama LLM to auto-generate prompt files in prompts/ folder.

The LLM reads actual spec file samples and produces optimized system prompts
tailored to the spec's format and content type.

Usage:
    # Generate compare prompt (reads spec samples to understand format)
    python scripts/run_generate_prompts.py --type compare --specs specs/

    # Generate analyze prompt (for QML module summarization)
    python scripts/run_generate_prompts.py --type analyze

    # Regenerate all prompts
    python scripts/run_generate_prompts.py --all --specs specs/
"""

import sys
import subprocess
import argparse
import random
from pathlib import Path

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
from utils.llm_client import LLMClient    # noqa: E402
from utils.logger import setup_logger     # noqa: E402

logger = setup_logger("GenPrompts", "GenPrompts_")


# ---------------------------------------------------------------------------
# Meta-prompts: instructions for the LLM to write a system prompt
# ---------------------------------------------------------------------------

METAPROMPT_COMPARE = """You are a prompt engineer. Your task is to write a SYSTEM PROMPT for an LLM.

The system prompt you write will be used to instruct an LLM to compare:
  - A section of a Japanese automotive instrument cluster specification document
  - QML source code from the implementation

The spec format you observed from the samples below contains:
{spec_observations}

Write a system prompt that instructs the LLM to:
1. Output ONLY a valid JSON object (no markdown, no explanations)
2. Compare the spec section against the QML code
3. Identify: implemented features, missing features, extra (over-implemented) features
4. Output format: {{"match_rate":"0-100%","implemented":[...],"missing":[...],"extra":[...],"notes":"..."}}
5. Be specific about what signals, states, and conditions to look for
6. notes field must be in Japanese

Make the system prompt specific to the automotive/CAN signal format observed in the specs.
Include concrete examples of what to match (CAN signal names, display states, EOL conditions).

Output ONLY the system prompt text. No meta-commentary, no markdown fences, no explanation."""


METAPROMPT_ANALYZE = """You are a prompt engineer. Your task is to write a SYSTEM PROMPT for an LLM.

The system prompt you write will be used to instruct an LLM to analyze QML source code modules.
The QML code is from an automotive instrument cluster application.

Write a system prompt that instructs the LLM to:
1. Output ONLY a valid JSON object (no markdown, no explanations)
2. Analyze the given QML module structure (functions, signals, properties, bindings, child components)
3. Determine the module's purpose and implementation status
4. Output format: {{"summary":"one-line description","status":"implemented|partial|stub|unknown","issues":[...]}}
5. status choices: "implemented" = fully working, "partial" = some features missing, "stub" = placeholder only, "unknown" = cannot determine
6. issues: list potential problems (empty list if none)

Output ONLY the system prompt text. No meta-commentary, no markdown fences, no explanation."""


METAPROMPT_EXTRACTOR = """You are a prompt engineer. Your task is to write a SYSTEM PROMPT for an LLM.

The system prompt you write will be used to instruct an LLM acting as a requirements analyst.
It will receive a section of a Japanese automotive instrument cluster specification document
and must extract each testable requirement from it.

The spec format contains behavior tables with:
{spec_observations}

Write a system prompt that instructs the LLM to:
1. Output ONLY a valid JSON object (no markdown, no explanations)
2. Extract each testable requirement (one per table row or condition)
3. Output format: {{"requirements": [{{"id": "REQ-001", "description": "one-line English summary", "inputs": {{"signal": "value"}}, "expected": "expected display state or behavior"}}]}}
4. Rules: id is sequential REQ-001/002/..., description in English, inputs as key-value pairs from input columns, expected from output column
5. Skip rows where all outputs are non-display/off with no specific condition
6. Maximum 20 requirements per section

Make the prompt specific to CAN signal table format used in automotive specs.
Output ONLY the system prompt text. No meta-commentary, no markdown fences, no explanation."""


METAPROMPT_JUDGE = """You are a prompt engineer. Your task is to write a SYSTEM PROMPT for an LLM.

The system prompt you write will be used to instruct an LLM acting as a code review expert
for automotive instrument cluster QML software.

It will receive:
1. A single requirement (id, description, inputs dict, expected output)
2. Relevant QML source code chunks retrieved from the codebase

Write a system prompt that instructs the LLM to:
1. Output ONLY a valid JSON object (no markdown, no explanations)
2. Determine if the requirement is implemented in the provided code
3. Output format: {{"req_id": "REQ-001", "status": "implemented|missing|partial", "confidence": "high|medium|low", "evidence": "one sentence"}}
4. Definitions:
   - implemented: code clearly handles this input/output condition
   - missing: no matching logic found
   - partial: related logic exists but condition or output is incomplete
   - confidence high: direct match (function name, signal name, property value)
   - confidence medium: indirect evidence (similar logic, related component)
   - confidence low: code chunks not clearly related
5. evidence: one concise sentence describing what was found or not found

Output ONLY the system prompt text. No meta-commentary, no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_spec_sections(specs_dir: Path, n_samples: int = 3) -> str:
    """Read a few spec files and extract representative sections."""
    spec_files = sorted(specs_dir.glob("*.md"))
    if not spec_files:
        return "(no spec files found)"

    sampled = random.sample(spec_files, min(n_samples, len(spec_files)))
    observations = []

    for sf in sampled:
        text = sf.read_text(encoding="utf-8", errors="replace")
        # Extract first 1000 chars to show format
        preview = text[:1000].strip()
        observations.append(f"--- {sf.name} ---\n{preview}\n")

    return "\n".join(observations)


def generate_prompt(llm: LLMClient, metaprompt: str) -> str:
    """Ask LLM to generate a system prompt."""
    messages = [
        {"role": "user", "content": metaprompt},
    ]
    return llm.chat(messages, stream=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_llm() -> LLMClient:
    models = load_yaml(SCRIPT_DIR / "config" / "models.yaml")
    default = models["default_model"]
    cfg = models["models"][default]
    return LLMClient(
        base_url=cfg["api_base"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=0.4,  # slightly higher for creative prompt writing
        max_tokens=1500,
    )


def cmd_compare(specs_dir: Path, output_path: Path):
    logger.info(f"[COMPARE] Sampling specs from {specs_dir} ...")
    spec_observations = sample_spec_sections(specs_dir, n_samples=3)

    metaprompt = METAPROMPT_COMPARE.format(spec_observations=spec_observations)

    llm = get_llm()
    logger.info("[LLM] Generating compare system prompt ...")
    generated = llm.chat([{"role": "user", "content": metaprompt}], stream=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated.strip(), encoding="utf-8")
    logger.info(f"[SAVE] {output_path}")
    print(f"\n--- Generated compare prompt saved to {output_path} ---")
    print(generated[:500] + "..." if len(generated) > 500 else generated)


def cmd_analyze(output_path: Path):
    llm = get_llm()
    logger.info("[LLM] Generating analyze system prompt ...")
    generated = llm.chat([{"role": "user", "content": METAPROMPT_ANALYZE}], stream=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated.strip(), encoding="utf-8")
    logger.info(f"[SAVE] {output_path}")
    print(f"\n--- Generated analyze prompt saved to {output_path} ---")
    print(generated[:500] + "..." if len(generated) > 500 else generated)


def cmd_extractor(specs_dir: Path, output_path: Path):
    logger.info(f"[EXTRACTOR] Sampling specs from {specs_dir} ...")
    spec_observations = sample_spec_sections(specs_dir, n_samples=3)
    metaprompt = METAPROMPT_EXTRACTOR.format(spec_observations=spec_observations)

    llm = get_llm()
    logger.info("[LLM] Generating extractor system prompt ...")
    generated = llm.chat([{"role": "user", "content": metaprompt}], stream=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated.strip(), encoding="utf-8")
    logger.info(f"[SAVE] {output_path}")
    print(f"\n--- Generated extractor prompt saved to {output_path} ---")
    print(generated[:500] + "..." if len(generated) > 500 else generated)


def cmd_judge(output_path: Path):
    llm = get_llm()
    logger.info("[LLM] Generating judge system prompt ...")
    generated = llm.chat([{"role": "user", "content": METAPROMPT_JUDGE}], stream=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated.strip(), encoding="utf-8")
    logger.info(f"[SAVE] {output_path}")
    print(f"\n--- Generated judge prompt saved to {output_path} ---")
    print(generated[:500] + "..." if len(generated) > 500 else generated)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Use LLM to generate system prompts for the analysis pipeline"
    )
    parser.add_argument("--type", choices=["compare", "analyze", "extractor", "judge"],
                        default=None, help="Which prompt to generate")
    parser.add_argument("--all", action="store_true", help="Generate all prompts")
    parser.add_argument("--specs", default="specs",
                        help="Path to specs folder (used for compare/extractor prompts, default: specs/)")
    args = parser.parse_args()

    if not args.type and not args.all:
        parser.print_help()
        sys.exit(1)

    specs_dir = SCRIPT_DIR / args.specs
    prompts_dir = SCRIPT_DIR / "prompts"

    generate_compare   = args.all or args.type == "compare"
    generate_analyze   = args.all or args.type == "analyze"
    generate_extractor = args.all or args.type == "extractor"
    generate_judge     = args.all or args.type == "judge"

    if generate_compare:
        cmd_compare(specs_dir, prompts_dir / "compare.md")

    if generate_analyze:
        cmd_analyze(prompts_dir / "analyze.md")

    if generate_extractor:
        cmd_extractor(specs_dir, prompts_dir / "extractor.md")

    if generate_judge:
        cmd_judge(prompts_dir / "judge.md")


if __name__ == "__main__":
    main()

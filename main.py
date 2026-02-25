# main.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import yaml

from utils.install_model import run_model
from utils.llm_client import LLMClient
from utils.project_utils import summarize_project
from utils.diff_utils import is_unified_diff, apply_unified_diff

from agents.generator_agent import GenerateAgent
from agents.linter_agent import LinterAgent
from agents.reviewer_agent import ReviewerAgent


def load_yaml(p):
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


SCRIPT_DIR = Path(__file__).resolve().parent

def main():
    # -------------------------------
    # 1) Load configs (기존 방식 그대로)
    # -------------------------------
    config = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
    prompts = load_yaml(SCRIPT_DIR / "config" / "prompts.yaml")
    models  = load_yaml(SCRIPT_DIR / "config" / "models.yaml")

    default = models["default_model"]
    mc = models["models"][default]

    # -------------------------------
    # 2) Ollama 환경 자동 준비 (기존 그대로)
    # -------------------------------
    run_model(
        pull_model=mc["model"],
        model_dir=Path(config["ollama"]["model_dir"]),
        check_only=False
    )

    # -------------------------------
    # 3) LLM Client 준비
    # -------------------------------
    llm = LLMClient(
        base_url=mc["api_base"],
        api_key=mc["api_key"],
        model=mc["model"],
        temperature=mc.get("temperature", 0.2)
    )

    # -------------------------------
    # 4) Summarize (기존 방식 그대로)
    # -------------------------------
    ctx = summarize_project(
        ".",
        tuple(config["project"]["target_exts"]),
        max_chars=9000
    )

    # -------------------------------
    # 5) Generate Agent
    # -------------------------------
    gen_agent = GenerateAgent(llm, prompts)
    gen = gen_agent.generate(ctx)
    print("=== GENERATE ===")
    print(gen)

    if not is_unified_diff(gen):
        print("[ERR] not unified diff")
        return

    # -------------------------------
    # 6) Linter Agent (qmllint + static fix)
    # -------------------------------
    lint_agent = LinterAgent(llm, prompts, config)
    lint = lint_agent.apply_and_lint(gen)
    print("\n=== LINT ===")
    print(lint)

    static = lint_agent.static_fix(lint)
    print("\n=== STATIC FIX ===")
    print(static)

    # -------------------------------
    # 7) Reviewer Agent
    # -------------------------------
    reviewer = ReviewerAgent(llm, prompts)
    final = reviewer.review(gen, lint, static)
    print("\n=== FINAL PATCH ===")
    print(final)

    # -------------------------------
    # 8) Optionally apply patch
    # -------------------------------
    if config["options"]["apply_patch"] and is_unified_diff(final):
        applied = apply_unified_diff(final)
        if applied.get("applied"):
            os.system("git add -A")
            os.system('git commit -m "agent: apply final QML patch"')
            print("[OK] committed")
        else:
            print("[ERR] apply failed:", applied)


if __name__ == "__main__":
    main()
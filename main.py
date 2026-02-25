# main.py
# -*- coding: utf-8 -*-

import os

from pathlib import Path

from utils.install_model import run_model, wait_for_server
from utils.llm_client import LLMClient
from utils.project_utils import summarize_project, load_yaml
from utils.diff_utils import is_unified_diff, apply_unified_diff
from utils.parallel import run_agents_in_parallel
from utils.logger import setup_logger

from agents.pipeline import AgentPipeline
from agents.reviewer_agent import ReviewerAgent

SCRIPT_DIR = Path(__file__).resolve().parent
logger = setup_logger("Main", "Main_")

def main():

    # Load configs
    config  = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
    prompts = load_yaml(SCRIPT_DIR / "config" / "prompts.yaml")
    models  = load_yaml(SCRIPT_DIR / "config" / "models.yaml")

    default = models["default_model"]
    mc = models["models"][default]

    # Setting Ollama (example)
    run_model(
        pull_model=mc["model"],
        model_dir=Path(config["ollama"]["model_dir"]),
        check_only=False
    )
    
    logger.info("Waiting for Ollama server before LLMClient init...")
    
    if not wait_for_server(timeout_sec=30):
        logger.error("Ollama server not responding after startup.")
        return

    ctx = summarize_project(
        ".",
        tuple(config["project"]["target_exts"]),
        max_chars=9000
    )

    # Create multiple pipelines with slightly different temperatures for diversity
    temps = [
        mc.get("temperature", 0.2),
        min(mc.get("temperature", 0.2) + 0.1, 1.0),
        min(mc.get("temperature", 0.2) + 0.2, 1.0),
    ]

    model_cfgs = []
    for t in temps:
        cfg = dict(mc)
        cfg["temperature"] = t
        model_cfgs.append(cfg)

    pipelines = [
        AgentPipeline(agent_id=i, project_ctx=ctx, prompts=prompts, config=config, model_cfg=model_cfgs[i])
        for i in range(3)
    ]

    # Run pipelines in parallel
    logger.info("Running 3 agents in parallel...")
    results = run_agents_in_parallel(pipelines, max_workers=3)
    logger.info("All agents finished.")

    logger.info("\n=== INDIVIDUAL RESULTS ===")
    for r in results:
        if "error" in r and r["error"]:
            logger.error(f"[AGENT {r['id']}] ERROR: {r['error']}")
        else:
            logger.info(f"\n[AGENT {r['id']}]")
            logger.info("GEN:\n" + r["gen"][:6000])
            logger.info("\nLINT:\n" + r["lint"][:4000])
            logger.info("\nSTATIC:\n" + r["static"][:4000])
            logger.info("\nFINAL:\n" + r["final"][:6000])

    # Reviewer agent for final aggregation
    # Use a fresh LLM client for the reviewer to avoid any potential state issues
    agg_llm = LLMClient(
        base_url=mc["api_base"],
        api_key=mc["api_key"],
        model=mc["model"],
        temperature=mc.get("temperature", 0.2)
    )
    final_reviewer = ReviewerAgent(agg_llm, prompts)
    final_patch = final_reviewer.final_decision(results)

    logger.info("\n=== FINAL PATCH (AGGREGATED) ===")
    logger.info(final_patch if final_patch else "[EMPTY]")

    # Optionally apply patch
    if config["options"]["apply_patch"] and is_unified_diff(final_patch):
        applied = apply_unified_diff(final_patch)
        if applied.get("applied"):
            os.system("git add -A")
            os.system('git commit -m "agent: apply final QML patch (3-agent ensemble)"')
            logger.info("[OK] committed")
        else:
            logger.error("[ERR] apply failed:", applied)

if __name__ == "__main__":
    main()
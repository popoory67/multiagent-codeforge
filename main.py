# main.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from agents.generator_agent import GenerateAgent
from utils.install_model import run_model, wait_for_server
from utils.llm_client import LLMClient
from utils.project_utils import summarize_project, load_yaml
from utils.diff_utils import is_unified_diff, apply_unified_diff
from utils.parallel import run_agents_in_parallel
from utils.logger import setup_logger

from agents.pipeline import AgentPipeline
from agents.reviewer_agent import ReviewerAgent


class GenerationJob:
    """
    Reuse run_agents_in_parallel by exposing a .run() method.
    Builds a synchronous LLMClient with a specific temperature
    and returns {"id": int, "gen": str}.
    """
    def __init__(self, agent_id, temperature, model_cfg, prompts, project_summary, logger):
        self.id = agent_id
        self.temperature = temperature
        self.model_cfg = model_cfg
        self.prompts = prompts
        self.project_summary = project_summary
        self.logger = logger

    def run(self) -> dict:
        cfg = dict(self.model_cfg)
        cfg["temperature"] = self.temperature

        llm = LLMClient(
            base_url=cfg["api_base"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            temperature=cfg["temperature"],
        )

        self.logger.info(f"[SEQ GEN] Agent {self.id} temperature={cfg['temperature']}")
        gen_agent = GenerateAgent(llm, self.prompts)
        gen_patch = gen_agent.generate(self.project_summary)

        return {"id": self.id, "gen": gen_patch}


SCRIPT_DIR = Path(__file__).resolve().parent
logger = setup_logger("Main", "Main_")

def main():

    # Load configs
    config  = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
    prompts = load_yaml(SCRIPT_DIR / "config" / "prompts.yaml")
    models  = load_yaml(SCRIPT_DIR / "config" / "models.yaml")

    # Select model config
    default = models["default_model"]
    model_cfg = models["models"][default]

    # Setting Ollama (example)
    run_model(
        pull_model=model_cfg["model"],
        model_dir=Path(config["ollama"]["model_dir"]),
        check_only=False
    )
    
    logger.info("Waiting for Ollama server before LLMClient init...")
    if not wait_for_server(timeout_sec=30):
        logger.error("Ollama server not responding after startup.")
        return

    # Project summary
    project_summary = summarize_project(
        ".",
        tuple(config["project"]["target_exts"]),
        max_chars=9000
    )

    # Create multiple pipelines with slightly different temperatures for diversity
    base_temp = model_cfg.get("temperature", 0.2)

    # Temperature diversification (clamped to [0.0, 1.0])
    temps = [
        max(0.0, min(base_temp, 1.0)),
        max(0.0, min(base_temp + 0.1, 1.0)),
        max(0.0, min(base_temp + 0.2, 1.0)),
    ]

    # --- Run generator phase first (parallel via your existing helper) ---
    gen_jobs = [
        GenerationJob(i, temps[i], model_cfg, prompts, project_summary, logger)
        for i in range(3)
    ]
    gen_results = run_agents_in_parallel(gen_jobs, max_workers=3)

    gens = [None] * 3
    for r in gen_results:
        if r.get("error"):
            logger.error(f"[GEN {r.get('id')}] ERROR: {r['error']}")
            return
        gens[r["id"]] = r["gen"]

    # Sanity check
    if any(g is None for g in gens):
        missing = [i for i, g in enumerate(gens) if g is None]
        logger.error(f"Some generated patches are missing at indices: {missing}")
        return

    # Build pipelines (each agent gets its own generated patch)
    pipelines = [
        AgentPipeline(
            agent_id=i,
            prompts=prompts,
            config=config,
            model_cfg=model_cfg, # shared model_cfg (generator only use temp diversify)
            generated_patch=gens[i], # unique generated patch from above
        )
        for i in range(3)
    ]

    # Run pipelines in parallel
    logger.info("Running 3 agents in parallel...")
    results = run_agents_in_parallel(pipelines, max_workers=3)
    logger.info("All agents finished.")

    logger.info("\n=== INDIVIDUAL RESULTS ===")
    for r in results:
        if r.get("error"):
            logger.error(f"[AGENT {r['id']}] ERROR: {r['error']}")
        else:
            logger.info(f"\n[AGENT {r['id']}]")
            logger.info("GEN:\n" + (r.get("gen","")[:6000]))
            logger.info("\nLINT:\n" + (r.get("lint","")[:4000]))
            logger.info("\nSTATIC:\n" + (r.get("static","")[:4000]))
            logger.info("\nFINAL:\n" + (r.get("final","")[:6000]))

    # Reviewer agent for final aggregation
    agg_llm = LLMClient(
        base_url=model_cfg["api_base"],
        api_key=model_cfg["api_key"],
        model=model_cfg["model"],
        temperature=model_cfg.get("temperature", 0.2),
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
            logger.error(f"[ERR] apply failed: {applied}")


if __name__ == "__main__":
    main()
# scripts/run_all.py
import sys
import subprocess
import asyncio
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(SCRIPT_DIR))

from utils.logger import setup_logger
from utils.project_utils import summarize_project, load_yaml
from utils.diff_utils import is_unified_diff, apply_unified_diff
from utils.llm_client import LLMClient
from utils.async_executor import run_async_jobs
from utils.generate_job import GenerationJob, PipelineJob
from agents.reviewer_agent import ReviewerAgent

logger = setup_logger("Main", "Main_")

async def run_full_pipeline():

    # Load configs
    config  = load_yaml(SCRIPT_DIR / "config" / "config.yaml")
    prompts = load_yaml(SCRIPT_DIR / "config" / "prompts.yaml")
    models  = load_yaml(SCRIPT_DIR / "config" / "models.yaml")

    default = models["default_model"]
    model_cfg = models["models"][default]

    # Project summary
    project_summary = summarize_project(
        ".",
        tuple(config["project"]["target_exts"]),
        max_chars=9000
    )

    base_temp = model_cfg.get("temperature", 0.2)
    temps = [base_temp, base_temp + 0.1, base_temp + 0.2]

    # ---- GENERATOR PHASE (ASYNC PARALLEL) ----
    gen_jobs = [
        GenerationJob(i, temps[i], model_cfg, prompts, project_summary, logger)
        for i in range(3)
    ]

    gen_results = await run_async_jobs(gen_jobs, workers=3)

    gens = [None] * 3
    for r in gen_results:
        if r.get("error"):
            logger.error(f"[GEN {r['id']}] ERROR: {r['error']}")
            return
        gens[r["id"]] = r["gen"]

    # ---- PIPELINE PHASE (ASYNC PARALLEL) ----
    pipelines = [
        PipelineJob(
            agent_id=i,
            prompts=prompts,
            config=config,
            model_cfg=model_cfg,
            generated_patch=gens[i],
            project_summary=project_summary,
            logger=logger
        )
        for i in range(3)
    ]

    results = await run_async_jobs(pipelines, workers=3)

    # ---- AGGREGATE ----
    agg_llm = LLMClient(
        base_url=model_cfg["api_base"],
        api_key=model_cfg["api_key"],
        model=model_cfg["model"],
        temperature=model_cfg.get("temperature", 0.2),
    )
    final_reviewer = ReviewerAgent(agg_llm, prompts)
    final_patch = final_reviewer.final_decision(results)

    logger.info("\n=== FINAL PATCH ===")
    logger.info(final_patch)

    # Apply patch
    if config["options"]["apply_patch"] and is_unified_diff(final_patch):
        applied = apply_unified_diff(final_patch)
        if applied.get("applied"):
            add_result = subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
            if add_result.returncode != 0:
                logger.error(f"[ERR] git add failed: {add_result.stderr}")
                return
            commit_result = subprocess.run(
                ["git", "commit", "-m", "agent: apply final QML patch (3-agent ensemble)"],
                capture_output=True, text=True
            )
            if commit_result.returncode != 0:
                logger.error(f"[ERR] git commit failed: {commit_result.stderr}")
                return
            logger.info("[OK] committed")
        else:
            logger.error(f"[ERR] apply failed: {applied}")

if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
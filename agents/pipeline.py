# agents/pipeline.py
# -*- coding: utf-8 -*-

from utils.llm_client import LLMClient
from agents.generator_agent import GenerateAgent
from agents.linter_agent import LinterAgent
from agents.reviewer_agent import ReviewerAgent
from utils.logger import setup_logger

logger = setup_logger("Pipeline", "Pipeline_")

class AgentPipeline:
    def __init__(
            self,
            agent_id: int,
            prompts: dict,
            config: dict,
            model_cfg: dict,
            generated_patch: str,
            project_ctx: dict,
            logger=None,
        ):

        self.id = agent_id
        self.logger = logger or setup_logger(f"Pipeline-{agent_id}", f"Pipeline_{agent_id}_")
        self.prompts = prompts
        self.config = config
        self.model_cfg = model_cfg
        self.project_ctx = project_ctx

        # LLMClient setup (shared across all agents in the pipeline)
        self.llm = LLMClient(
            base_url=model_cfg["api_base"],
            api_key=model_cfg["api_key"],
            model=model_cfg["model"],
            temperature=model_cfg.get("temperature", 0.2),
        )

        self.generated_patch = generated_patch

        self.gen_agent = GenerateAgent(
            llm=self.llm,
            prompts=prompts,
            project_ctx=project_ctx,
            config=config,
            agent_id=f"GEN-{agent_id}"
        )
        self.gen_agent.logger = self.logger

        self.lint_agent = LinterAgent(
            llm=self.llm,
            prompts=prompts,
            project_ctx=project_ctx,
            config=config,
            agent_id=f"LINT-{agent_id}"
        )
        self.lint_agent.logger = self.logger

        self.reviewer_agent = ReviewerAgent(
            llm=self.llm,
            prompts=prompts,
            project_ctx=project_ctx,
            config=config,
            agent_id=f"REVIEW-{agent_id}"
        )
        self.reviewer_agent.logger = self.logger
        
    def run(self) -> dict:
        self.logger.info(f"[AGENT {self.id}] Starting pipeline")

        gen = self.generated_patch
        self.logger.info(f"[AGENT {self.id}] Step 1: Using pre-generated patch")

        self.logger.info(f"[AGENT {self.id}] Step 2: Running qmllint on generated patch...")
        lint = self.lint_agent.apply_and_lint(gen)

        self.logger.info(f"[AGENT {self.id}] Step 3: Static fix...")
        static = self.lint_agent.static_fix(lint)

        self.logger.info(f"[AGENT {self.id}] Step 4: Reviewer pass...")
        final = self.reviewer_agent.review(gen, lint, static)

        self.logger.info(f"[AGENT {self.id}] Finished.")
        self.logger.info("-" * 80)
    
        return {
            "id": self.id,
            "gen": gen,
            "lint": lint,
            "static": static,
            "final": final,
            "error": None,
        }
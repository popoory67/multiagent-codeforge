# agents/pipeline.py
# -*- coding: utf-8 -*-

from utils.llm_client import LLMClient
from agents.generator_agent import GenerateAgent
from agents.linter_agent import LinterAgent
from agents.reviewer_agent import ReviewerAgent
from utils.logger import setup_logger

logger = setup_logger("Pipeline", "Pipeline_")

class AgentPipeline:
    def __init__(self, agent_id: int, project_ctx: str, prompts: dict, config: dict, model_cfg: dict):
        self.id = agent_id
        self.project_ctx = project_ctx
        self.prompts = prompts
        self.config = config

        self.llm = LLMClient(
            base_url=model_cfg["api_base"],
            api_key=model_cfg["api_key"],
            model=model_cfg["model"],
            temperature=model_cfg.get("temperature", 0.2),
        )

        self.gen_agent = GenerateAgent(self.llm, prompts)
        self.lint_agent = LinterAgent(self.llm, prompts, config)
        self.reviewer = ReviewerAgent(self.llm, prompts)
        self.logger = logger
    
    def run(self) -> dict:
            self.logger.info("Starting pipeline", extra={"agent_id": self.id})

            self.logger.info(f"[AGENT {self.id}] Step 1: Generating patch...")
            gen = self.gen_agent.generate(self.project_ctx)

            self.logger.info(f"[AGENT {self.id}] Step 2: Running qmllint on generated patch...")
            lint = self.lint_agent.apply_and_lint(gen)

            self.logger.info(f"[AGENT {self.id}] Step 3: Static fix...")
            static = self.lint_agent.static_fix(lint)

            self.logger.info(f"[AGENT {self.id}] Step 4: Reviewer pass...")
            final = self.reviewer.review(gen, lint, static)

            self.logger.info(f"[AGENT {self.id}] Finished.")
            self.logger.info("-" * 80)
        
            return {
                "id": self.id,
                "gen": gen,
                "lint": lint,
                "static": static,
                "final": final,
            }

# utils/generate_job.py
from utils.llm_client import LLMClient
from agents.generator_agent import GenerateAgent
from agents.pipeline import AgentPipeline

class GenerationJob:
    def __init__(self, agent_id, temperature, model_cfg, prompts, project_summary, logger):
        self.id = agent_id
        self.temperature = temperature
        self.model_cfg = model_cfg
        self.prompts = prompts
        self.project_summary = project_summary
        self.logger = logger

    def run(self):
        cfg = dict(self.model_cfg)
        cfg["temperature"] = self.temperature

        llm = LLMClient(
            base_url=cfg["api_base"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            logger=self.logger,
            agent_id=self.id,
        )

        gen_agent = GenerateAgent(llm, self.prompts)
        patch = gen_agent.generate(self.project_summary)

        return {"id": self.id, "gen": patch, "error": None}

class PipelineJob:
    def __init__(self, agent_id, prompts, config, model_cfg, generated_patch, project_summary, logger):
        self.id = agent_id
        self.prompts = prompts
        self.config = config
        self.model_cfg = model_cfg
        self.generated_patch = generated_patch
        self.project_summary = project_summary
        self.logger = logger

    def run(self):
        pipeline = AgentPipeline(
            agent_id=self.id,
            prompts=self.prompts,
            config=self.config,
            model_cfg=self.model_cfg,
            generated_patch=self.generated_patch,
            project_ctx={
                "summary": self.project_summary
            },
            logger=self.logger
        )

        return pipeline.run()
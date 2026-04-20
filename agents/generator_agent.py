# agents/generate_agent.py
# -*- coding: utf-8 -*-
from agents.base_agent import BaseAgent
from utils.diff_utils import is_unified_diff, normalize_unified_diff

class GenerateAgent(BaseAgent):

    def generate(self, project_ctx=None) -> str:
        project_ctx = project_ctx or self.project_ctx

        task = self.prompts["qmlcoder"]["task_description"]
        system = self.prompts["qmlcoder"]["system"]

        user = (
            f"{task}\n\n"
            f"Project context:\n{project_ctx}\n\n"
            "Output unified diff ONLY. No explanations."
        )

        raw = self.chat(system, user, stream_log=True, log_lines=False, batch_size=20)
        diff = normalize_unified_diff(raw)
        return diff if is_unified_diff(diff) else ""
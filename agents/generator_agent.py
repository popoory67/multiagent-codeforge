# agents/generate_agent.py
from utils.diff_utils import normalize_unified_diff, is_unified_diff

class GenerateAgent:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def generate(self, project_ctx):
        task = self.prompts["qmlcoder"]["task_description"]
        system = self.prompts["qmlcoder"]["system"]

        prompt_user = (
            f"{task}\n\n"
            f"프로젝트 컨텍스트:\n{project_ctx}\n\n"
            "출력은 unified diff ONLY. 설명 금지."
        )

        raw = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_user},
        ])

        diff = normalize_unified_diff(raw)
        return diff if is_unified_diff(diff) else ""
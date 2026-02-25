# agents/reviewer_agent.py
from utils.diff_utils import normalize_unified_diff, is_unified_diff

class ReviewerAgent:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def review(self, gen, lint, static):
        fit = self.prompts["qmlcoder"]["fitness_criteria"]

        prompt = (
            f"[FITNESS]\n{fit}\n\n"
            f"[GEN]\n{gen}\n\n"
            f"[LINT]\n{lint}\n\n"
            f"[STATIC]\n{static}\n\n"
            "최종 승인된 unified diff만 출력."
        )

        raw = self.llm.chat([
            {"role": "system", "content": self.prompts["reviewer"]["system"]},
            {"role": "user", "content": prompt},
        ])

        patch = normalize_unified_diff(raw)
        return patch if is_unified_diff(patch) else ""
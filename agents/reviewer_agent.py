# agents/reviewer_agent.py
from agents.base_agent import BaseAgent
from utils.diff_utils import normalize_unified_diff, is_unified_diff

class ReviewerAgent(BaseAgent):

    def review(self, gen, lint, static):
        fit = self.prompts["qmlcoder"]["fitness_criteria"]
        system = self.prompts["reviewer"]["system"]

        user = (
            f"[FITNESS]\n{fit}\n\n"
            f"[GEN]\n{gen}\n\n"
            f"[LINT]\n{lint}\n\n"
            f"[STATIC]\n{static}\n\n"
            "최종 승인된 unified diff만 출력."
        )

        raw = self.chat(system, user)
        patch = normalize_unified_diff(raw)
        return patch if is_unified_diff(patch) else ""

    def final_decision(self, agent_results: list[dict]):
        system = self.prompts["reviewer"]["system"]
        fit = self.prompts["qmlcoder"]["fitness_criteria"]

        body = []
        for r in agent_results:
            body.append(
                f"[AGENT {r['id']}]\n"
                f"GEN:\n{r.get('gen','')}\n\n"
                f"LINT:\n{r.get('lint','')}\n\n"
                f"STATIC:\n{r.get('static','')}\n\n"
                f"FINAL:\n{r.get('final','')}\n"
            )

        user = (
            f"[FITNESS]\n{fit}\n\n"
            + "\n".join(body)
            + "\n위 결과 중 품질/정합성이 가장 높은 하나의 unified diff만 출력."
        )

        raw = self.chat(system, user, stream_log=True, log_lines=False, batch_size=10)
        final = normalize_unified_diff(raw)
        return final if is_unified_diff(final) else ""
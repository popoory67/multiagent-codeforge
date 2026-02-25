# agents/reviewer_agent.py
# -*- coding: utf-8 -*-

from utils.diff_utils import normalize_unified_diff, is_unified_diff

class ReviewerAgent:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def review(self, gen, lint, static):
        fit = self.prompts["qmlcoder"]["fitness_criteria"]
        system = self.prompts["reviewer"]["system"]

        prompt = (
            f"[FITNESS]\n{fit}\n\n"
            f"[GEN]\n{gen}\n\n"
            f"[LINT]\n{lint}\n\n"
            f"[STATIC]\n{static}\n\n"
            "최종 승인된 unified diff만 출력."
        )

        raw = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])

        patch = normalize_unified_diff(raw)
        return patch if is_unified_diff(patch) else ""

    # 집계 Reviewer용 메서드 (3개 결과 비교)
    def final_decision(self, agent_results: list[dict]):
        system = self.prompts["reviewer"]["system"]
        fit = self.prompts["qmlcoder"]["fitness_criteria"]

        # 각 에이전트 결과를 포맷팅하여 하나의 프롬프트로 집계
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

        raw = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        final = normalize_unified_diff(raw)
        return final if is_unified_diff(final) else ""
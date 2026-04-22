# agents/judge_agent.py
"""
Agent 3 — Judge

Given a single requirement and relevant QML code chunks retrieved by the Searcher,
determines whether the requirement is implemented.

Output JSON:
{
  "req_id": "REQ-001",
  "status": "implemented" | "missing" | "partial",
  "confidence": "high" | "medium" | "low",
  "evidence": "brief description of where/why"
}
"""

import json
from agents.base_agent import BaseAgent

_DEFAULT_SYSTEM = """You are a code review expert for automotive instrument cluster QML software.

You will be given:
1. A single requirement (id, description, expected inputs/outputs)
2. Relevant QML source code chunks retrieved from the codebase

Determine if the requirement is implemented in the code.

Output ONLY a valid JSON object (no markdown, no explanations):
{
  "req_id": "REQ-001",
  "status": "implemented" | "missing" | "partial",
  "confidence": "high" | "medium" | "low",
  "evidence": "one sentence: what you found or did not find in the code"
}

Definitions:
- implemented: the code clearly handles this input/output condition
- missing: no matching logic found in the code
- partial: related logic exists but condition or output is incomplete
- confidence high: direct match found (function name, signal name, property value)
- confidence medium: indirect evidence (similar logic, related component)
- confidence low: code chunks are not clearly related

Output valid JSON ONLY."""


class JudgeAgent(BaseAgent):
    """Judges whether a single requirement is implemented in the given code."""

    def judge(self, requirement: dict, code_chunks: list[dict]) -> dict:
        """
        Args:
            requirement: dict with id, description, inputs, expected
            code_chunks: list of RAG hits with text and source

        Returns:
            dict with req_id, status, confidence, evidence
        """
        system = self.prompts.get("judge", {}).get("system", _DEFAULT_SYSTEM)

        # Format code context (keep within context window)
        code_text = "\n\n".join(
            f"[{c.get('source', '?')}]\n{c.get('text', '')}"
            for c in code_chunks[:4]
        )[:2000]

        req_text = json.dumps(requirement, ensure_ascii=False, indent=2)

        user = (
            f"## Requirement\n{req_text}\n\n"
            f"## Relevant Code\n{code_text}"
        )

        raw = self.chat(system, user, stream=False)

        try:
            result = json.loads(raw)
            result.setdefault("req_id", requirement.get("id", "?"))
            result.setdefault("status", "missing")
            result.setdefault("confidence", "low")
            result.setdefault("evidence", "")
            if self.logger:
                self.logger.info(
                    f"[Judge] {result['req_id']} → {result['status']} "
                    f"({result['confidence']})"
                )
            return result
        except (json.JSONDecodeError, AttributeError):
            if self.logger:
                self.logger.warning(
                    f"[Judge] JSON parse failed for {requirement.get('id')}: {raw[:100]}"
                )
            return {
                "req_id": requirement.get("id", "?"),
                "status": "missing",
                "confidence": "low",
                "evidence": f"parse error: {raw[:80]}",
            }

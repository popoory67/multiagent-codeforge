# agents/extractor_agent.py
"""
Agent 1 — Extractor

Reads a spec section (behavior table) and extracts structured requirements.
Each requirement is a single testable condition: given inputs → expected output.

Output JSON:
{
  "requirements": [
    {
      "id": "REQ-001",
      "description": "brief description",
      "inputs": {"signal": "value", ...},
      "expected": "expected display state or action"
    },
    ...
  ]
}
"""

import json
from agents.base_agent import BaseAgent

_DEFAULT_SYSTEM = """You are a requirements analyst for automotive instrument cluster software.

Given a section of a Japanese specification document, extract each testable requirement.
A requirement is one row or condition in a behavior table: given specific inputs → expected output.

Output ONLY a valid JSON object (no markdown, no explanations):
{
  "requirements": [
    {
      "id": "REQ-001",
      "description": "one-line description in English",
      "inputs": {"IG": "ON", "signal_name": "value"},
      "expected": "expected display state or behavior"
    }
  ]
}

Rules:
- Each row in a behavior table = one requirement
- id: sequential REQ-001, REQ-002, ...
- description: concise English summary
- inputs: key-value pairs from the input columns
- expected: the output column value
- Skip rows where all outputs are "non-display" or "off" with no condition specifics
- Maximum 20 requirements per section
- Output valid JSON ONLY"""


class ExtractorAgent(BaseAgent):
    """Extracts structured requirements from a spec section."""

    def extract(self, section_heading: str, section_text: str) -> list[dict]:
        """
        Returns a list of requirement dicts, or empty list on failure.
        """
        system = self.prompts.get("extractor", {}).get("system", _DEFAULT_SYSTEM)

        # Keep within llama3.1:8b context: ~2500 chars for spec content
        content = section_text[:2500]
        user = f"Section: {section_heading}\n\n{content}"

        raw = self.chat(system, user, stream=False)

        try:
            result = json.loads(raw)
            reqs = result.get("requirements", [])
            if self.logger:
                self.logger.info(
                    f"[Extractor] {section_heading[:40]} → {len(reqs)} requirements"
                )
            return reqs
        except (json.JSONDecodeError, AttributeError):
            if self.logger:
                self.logger.warning(
                    f"[Extractor] JSON parse failed for '{section_heading[:40]}': {raw[:100]}"
                )
            return []

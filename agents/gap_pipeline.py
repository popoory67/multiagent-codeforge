# agents/gap_pipeline.py
"""
3-Agent Gap Analysis Pipeline

For each spec file:
  For each behavior section:
    Agent 1 (Extractor): section text → requirements list
    For each requirement:
      Agent 2 (Searcher): RAG retrieval → relevant QML code chunks  [no LLM]
      Agent 3 (Judge):    requirement + code → implemented/missing/partial

Aggregates all judgments into a per-spec gap report.
"""

import json
from pathlib import Path
from datetime import datetime

from agents.extractor_agent import ExtractorAgent
from agents.judge_agent import JudgeAgent
from utils.rag_store import RAGStore


class GapPipeline:
    """
    Orchestrates the 3-agent gap analysis for a single spec file.
    """

    def __init__(self, llm, prompts: dict, rag_store: RAGStore,
                 behavior_headings: list, skip_headings: list,
                 logger=None):
        self.extractor = ExtractorAgent(llm, prompts)
        self.judge = JudgeAgent(llm, prompts)
        self.extractor.logger = logger
        self.judge.logger = logger
        self.rag = rag_store
        self.behavior_headings = behavior_headings
        self.skip_headings = skip_headings
        self.logger = logger

    # ------------------------------------------------------------------
    # Agent 2: Searcher (RAG, no LLM)
    # ------------------------------------------------------------------

    def _search(self, requirement: dict, spec_name: str) -> list[dict]:
        """Retrieve relevant QML code chunks for a requirement."""
        # Build a focused query from requirement fields
        parts = [spec_name, requirement.get("description", "")]
        for v in requirement.get("inputs", {}).values():
            parts.append(str(v))
        parts.append(requirement.get("expected", ""))
        query = " ".join(p for p in parts if p)[:300]

        hits = []
        # Prefer actual QML source over analysis summaries
        try:
            hits = self.rag.query("qml_source", query, n_results=4)
        except Exception:
            pass
        if not hits:
            try:
                hits = self.rag.query("analysis", query, n_results=4)
            except Exception:
                pass
        return hits

    # ------------------------------------------------------------------
    # Section selector
    # ------------------------------------------------------------------

    def _select_sections(self, sections: list[dict]) -> list[dict]:
        """Return behavior sections; fall back to non-input sections."""
        behavior = [s for s in sections
                    if any(kw in s["heading"] for kw in self.behavior_headings)]
        if behavior:
            return behavior
        filtered = [s for s in sections
                    if not any(kw in s["heading"] for kw in self.skip_headings)]
        return filtered[:5] if filtered else sections[:5]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, spec_name: str, sections: list[dict]) -> dict:
        """
        Run the full 3-agent pipeline for one spec file.

        Returns a gap report dict:
        {
          "spec": str,
          "analyzed_at": str,
          "total_requirements": int,
          "implemented": int,
          "partial": int,
          "missing": int,
          "match_rate": str,
          "details": [
            {
              "section": str,
              "req_id": str,
              "description": str,
              "status": str,
              "confidence": str,
              "evidence": str
            }
          ]
        }
        """
        selected = self._select_sections(sections)

        if self.logger:
            self.logger.info(
                f"[GapPipeline] {spec_name}: {len(selected)} behavior sections"
            )

        all_details = []

        for sec in selected:
            heading = sec["heading"]

            # --- Agent 1: Extract requirements ---
            requirements = self.extractor.extract(heading, sec["content"])
            if not requirements:
                if self.logger:
                    self.logger.warning(
                        f"[GapPipeline] No requirements extracted from '{heading[:40]}'"
                    )
                continue

            if self.logger:
                self.logger.info(
                    f"[GapPipeline] Section '{heading[:40]}': "
                    f"{len(requirements)} requirements → judging..."
                )

            # --- Agent 2 + 3: Search + Judge per requirement ---
            for req in requirements:
                code_chunks = self._search(req, spec_name)   # Agent 2
                verdict = self.judge.judge(req, code_chunks)  # Agent 3

                all_details.append({
                    "section": heading,
                    "req_id": verdict.get("req_id", req.get("id", "?")),
                    "description": req.get("description", ""),
                    "inputs": req.get("inputs", {}),
                    "expected": req.get("expected", ""),
                    "status": verdict.get("status", "missing"),
                    "confidence": verdict.get("confidence", "low"),
                    "evidence": verdict.get("evidence", ""),
                })

        # --- Aggregate ---
        counts = {"implemented": 0, "partial": 0, "missing": 0}
        for d in all_details:
            s = d["status"]
            if s in counts:
                counts[s] += 1
            else:
                counts["missing"] += 1

        total = len(all_details)
        if total > 0:
            impl_score = counts["implemented"] + counts["partial"] * 0.5
            match_rate = f"{round(impl_score / total * 100)}%"
        else:
            match_rate = "0%"

        return {
            "spec": spec_name,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_requirements": total,
            "implemented": counts["implemented"],
            "partial": counts["partial"],
            "missing": counts["missing"],
            "match_rate": match_rate,
            "details": all_details,
        }


# ------------------------------------------------------------------
# Report renderers
# ------------------------------------------------------------------

def render_gap_md(report: dict) -> str:
    lines = [
        f"# Gap Report: {report['spec']}",
        f"",
        f"- **Analyzed**: {report['analyzed_at']}",
        f"- **Match Rate**: {report['match_rate']}",
        f"- **Total Requirements**: {report['total_requirements']}",
        f"- **Implemented**: {report['implemented']}",
        f"- **Partial**: {report['partial']}",
        f"- **Missing**: {report['missing']}",
        f"",
        f"## Requirements",
        f"",
        f"| ID | Description | Status | Confidence | Evidence |",
        f"|----|-------------|--------|------------|----------|",
    ]
    for d in report["details"]:
        status_icon = {"implemented": "✓", "partial": "△", "missing": "✗"}.get(
            d["status"], "?"
        )
        desc = d["description"][:50]
        evidence = d["evidence"][:60]
        lines.append(
            f"| {d['req_id']} | {desc} | {status_icon} {d['status']} "
            f"| {d['confidence']} | {evidence} |"
        )
    return "\n".join(lines)


def render_index_md(reports: list[dict], output_dir: Path) -> str:
    lines = [
        "# Gap Analysis Index",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Specs analyzed**: {len(reports)}",
        "",
        "| Spec | Match Rate | Implemented | Partial | Missing | Total |",
        "|------|-----------|-------------|---------|---------|-------|",
    ]
    for r in reports:
        name = r["spec"]
        link = f"[{name}]({name}.md)"
        lines.append(
            f"| {link} | {r['match_rate']} | {r['implemented']} "
            f"| {r['partial']} | {r['missing']} | {r['total_requirements']} |"
        )
    return "\n".join(lines)

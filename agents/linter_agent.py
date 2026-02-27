# agents/linter_agent.py
# -*- coding: utf-8 -*-

import tempfile
import shutil
import os
from pathlib import Path

from agents.base_agent import BaseAgent
from utils.diff_utils import apply_unified_diff
from utils.qml_utils import collect_qml_files, run_qmllint

class LinterAgent(BaseAgent):
    """
    Get QML diff onto a temporal directory
    Do the qmllint analysis there, then return the lint output.
    """

    def _copy_project_to_temp(self, temp_dir: str):
        for name in os.listdir("."):
            if name.startswith(".git"):
                continue
            src = Path(name)
            dst = Path(temp_dir) / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    def apply_and_lint(self, diff: str) -> str:
        temp_dir = tempfile.mkdtemp(prefix="qml_agent_")
        try:
            if self.logger:
                self.logger.info(f"[LinterAgent {self.agent_id}] Copying project to temp...")
            self._copy_project_to_temp(temp_dir)

            if self.logger:
                self.logger.info(f"[LinterAgent {self.agent_id}] Applying diff to temp...")
            applied = apply_unified_diff(diff, repo_root=temp_dir)

            qmllint_path = self.config["tools"]["qmllint_path"]

            target_root = temp_dir if applied.get("applied") else "."
            qml_files = collect_qml_files(target_root)

            if self.logger:
                self.logger.info(f"[LinterAgent {self.agent_id}] Running qmllint ({len(qml_files)} files)...")
            lint_output = run_qmllint(qml_files, qmllint_path)

            return lint_output

        except Exception as e:
            if self.logger:
                self.logger.error(f"[LinterAgent {self.agent_id}] ERROR in apply_and_lint: {e}")
            return f"[ERROR] apply_and_lint failed: {e}"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def static_fix(self, lint_output: str) -> str:
        system = self.prompts["static_analyzer"]["system"]
        user = (
            "다음은 qmllint의 출력입니다. 문제가 있는 부분만 "
            "정확하게 수정하는 unified diff를 생성하세요.\n\n"
            f"[QMLLINT]\n{lint_output}\n"
            "출력은 unified diff ONLY."
        )
        raw = self.chat(system, user, stream_log=False, log_lines=False)
        return raw
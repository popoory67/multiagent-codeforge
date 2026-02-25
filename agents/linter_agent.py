# agents/linter_agent.py
import tempfile
import shutil
import os
from utils.diff_utils import apply_unified_diff
from utils.qml_utils import collect_qml_files, run_qmllint

class LinterAgent:
    def __init__(self, llm, prompts, config):
        self.llm = llm
        self.prompts = prompts
        self.config = config

    def apply_and_lint(self, diff):
        temp = tempfile.mkdtemp(prefix="qml_agent_")

        try:
            for name in os.listdir("."):
                if name.startswith(".git"):
                    continue
                src = os.path.join(".", name)
                dst = os.path.join(temp, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            applied = apply_unified_diff(diff, repo_root=temp)
            qmllint_path = self.config["tools"]["qmllint_path"]

            if applied.get("applied"):
                qmls = collect_qml_files(temp)
            else:
                qmls = collect_qml_files(".")

            lint = run_qmllint(qmls, qmllint_path)
            return lint

        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def static_fix(self, lint_output):
        system = self.prompts["static_analyzer"]["system"]
        user = f"qmllint 출력입니다. 문제가 있는 부분만 고치는 diff를 출력:\n{lint_output}"

        raw = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return raw
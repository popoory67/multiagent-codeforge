# -*- coding: utf-8 -*-
"""
multi_agents_qml.py

리팩터링 버전:
- prompts.yaml, config.yaml, models.yaml로 설정/프롬프트 완전 분리
- 코드 안에는 로직만 존재
"""

import os
import shutil
import yaml
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict
from unidiff import PatchSet
from scripts.install_model import run_model
from autogen import AssistantAgent

############################################################
# 1) YAML 로드
############################################################

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

prompts = load_yaml("prompts/prompts.yaml")
config = load_yaml("config/config.yaml")
models = load_yaml("config/models.yaml")

# 모델 선택
default_model_name = models["default_model"]
model_cfg = models["models"][default_model_name]

MODEL_NAME = default_model_name
OPENAI_API_BASE = model_cfg["api_base"]
OPENAI_API_KEY = os.getenv(model_cfg["api_key"].replace("${", "").replace("}", ""), model_cfg["api_key"])
TEMPERATURE = model_cfg.get("temperature", 0.2)

# 로컬 설정
QMLLINT_PATH = config["tools"]["qmllint_path"]
TARGET_EXTS = tuple(config["project"]["target_exts"])
APPLY_PATCH = config["options"]["apply_patch"]

OLLAMA_MODEL_DIR = config["ollama"]["model_dir"]
if OLLAMA_MODEL_DIR.startswith("${") and OLLAMA_MODEL_DIR.endswith("}"):
    env_name = OLLAMA_MODEL_DIR[2:-1]
    OLLAMA_MODEL_DIR = os.getenv(env_name)

OLLAMA_BIN_DIR = config["ollama"]["bin_dir"]
if OLLAMA_BIN_DIR not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + OLLAMA_BIN_DIR

OLLAMA_MODEL_DIR = Path(OLLAMA_MODEL_DIR).expanduser().resolve()

############################################################
# 2) 유틸리티
############################################################

def run_cmd(cmd: List[str], cwd: str = ".") -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def summarize_project(root: str = ".", max_chars: int = 12000) -> str:
    buf, total = [], 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(TARGET_EXTS):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except:
                continue
            header = f"\n\n### FILE: {os.path.relpath(path, root)}\n"
            chunk = header + content
            if total + len(chunk) > max_chars:
                buf.append(header + content[:max_chars - total] + "\n... [TRUNCATED]")
                return "".join(buf)
            buf.append(chunk)
            total += len(chunk)
    return "".join(buf)

def apply_unified_diff(patch_text: str, repo_root: str = ".") -> Dict:
    try:
        PatchSet(patch_text.splitlines(True))
    except Exception as e:
        return {"applied": False, "error": f"Invalid patch: {e}"}

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name

    code, out, err = run_cmd(["git", "apply", "--whitespace=fix", "--reject", tmp_path], cwd=repo_root)
    os.unlink(tmp_path)
    return {"applied": code == 0, "stdout": out, "stderr": err}

def git_commit_all(msg):
    run_cmd(["git", "add", "-A"])
    code, out, err = run_cmd(["git", "commit", "-m", msg])
    return (code == 0), out + err

def run_qmllint(paths, module_paths=None):
    cmd = [QMLLINT_PATH]
    for mp in (module_paths or []):
        cmd += ["-I", mp]
    cmd += paths
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return out + ("\nERR:\n" + err if err else "")


############################################################
# 3) LLM 에이전트 정의
############################################################

llm_config = {
    "config_list": [
        {
            "model": MODEL_NAME,
            "api_type": "openai",
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_API_BASE,
            "temperature": TEMPERATURE,
        }
    ]
}

code_agent = AssistantAgent(
    name="QMLCoder",
    system_message=prompts["qmlcoder"]["system"],
    llm_config=llm_config,
)

static_agent = AssistantAgent(
    name="QMLStaticAnalyzer",
    system_message=prompts["static_analyzer"]["system"],
    llm_config=llm_config,
)

review_agent = AssistantAgent(
    name="Reviewer",
    system_message=prompts["reviewer"]["system"],
    llm_config=llm_config,
)

############################################################
# 4) 메인 파이프라인
############################################################
def main():

    TASK_DESCRIPTION = prompts["qmlcoder"]["task_description"]
    FITNESS_CRITERIA = prompts["qmlcoder"]["fitness_criteria"]

    project_context = summarize_project(".", max_chars=9000)

    # === 1단계: 코드 생성 ===
    gen_res = code_agent.run(
        f"{TASK_DESCRIPTION}\n\n"
        f"프로젝트 컨텍스트:\n{project_context}\n\n"
        f"위 기준에 따라 필요한 패치를 unified diff 형식으로 출력하세요."
    )
    gen_patch = (gen_res.messages or "").strip()

    print("\n=== CODE GENERATOR PATCH ===\n", gen_patch)

    # === 2단계: 임시 디렉토리에 패치 적용 후 qmllint 실행 ===
    temp_dir = tempfile.mkdtemp(prefix="qml_agent_")
    try:
        for name in os.listdir("."):
            if name.startswith(".git") or name == os.path.basename(temp_dir):
                continue
            src = os.path.join(".", name)
            dst = os.path.join(temp_dir, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        if gen_patch.startswith(("diff --git", "--- ")):
            apply_res = apply_unified_diff(gen_patch, repo_root=temp_dir)
        else:
            apply_res = {"applied": False, "error": "No diff returned"}

        if apply_res.get("applied"):
            qmllint_out = run_qmllint([temp_dir])
        else:
            qmllint_out = (
                "패치 적용 실패 → 원본 대상으로 qmllint 실행\n" +
                run_qmllint(["."])
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n=== QMLLINT RESULT ===\n", qmllint_out)

    # === 3단계: qmllint 기반 보완 패치 ===
    static_res = static_agent.run(
        "다음은 qmllint 결과입니다. 문제가 되는 부분을 수정하는 diff만 출력하세요.\n\n"
        f"{qmllint_out}\n"
    )
    static_patch = (static_res.messages or "").strip()
    print("\n=== STATIC ANALYZER PATCH ===\n", static_patch)

    # === 4단계: Reviewer 최종 패치 ===
    review_res = review_agent.run(
        f"[FITNESS_CRITERIA]\n{FITNESS_CRITERIA}\n\n"
        f"[GEN_PATCH]\n{gen_patch}\n\n"
        f"[QMLLINT]\n{qmllint_out}\n\n"
        f"[STATIC_PATCH]\n{static_patch}\n"
    )
    final_patch = (review_res.messages or "").strip()
    print("\n=== FINAL PATCH ===\n", final_patch)

if __name__ == "__main__":
    run_model(pull_model=MODEL_NAME, model_dir=OLLAMA_MODEL_DIR)
    main()
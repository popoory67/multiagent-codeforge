# -*- coding: utf-8 -*-
import os
import sys
import json
import yaml
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict
from unidiff import PatchSet
from autogen import AssistantAgent
from scripts.install_model import run_model  # 모델 준비(서버 확인/이미 있으면 pull skip)

# ============================================================
# 1) 경로/설정 로딩 (스크립트 기준 경로 안전)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS_PATH = SCRIPT_DIR / "prompts" / "prompts.yaml"
CONFIG_PATH  = SCRIPT_DIR / "config"  / "config.yaml"
MODELS_PATH  = SCRIPT_DIR / "config"  / "models.yaml"

for pth, name in [(PROMPTS_PATH, "prompts.yaml"),
                  (CONFIG_PATH, "config.yaml"),
                  (MODELS_PATH, "models.yaml")]:
    if not pth.exists():
        raise FileNotFoundError(f"[ERR] {name} not found at: {pth}")

prompts = load_yaml(PROMPTS_PATH)
config  = load_yaml(CONFIG_PATH)
models  = load_yaml(MODELS_PATH)

# ============================================================
# 2) 모델/경로 설정
# ============================================================

# 모델 선택
default_model_name = models["default_model"]
model_cfg = models["models"][default_model_name]

MODEL_NAME = model_cfg.get("model", default_model_name)
OPENAI_API_BASE = model_cfg["api_base"]
# api_key가 ${ENV_NAME} 식이면 환경변수 치환
api_key_cfg = model_cfg.get("api_key", "")
if api_key_cfg.startswith("${") and api_key_cfg.endswith("}"):
    env_name = api_key_cfg[2:-1]
    OPENAI_API_KEY = os.getenv(env_name, "")
else:
    OPENAI_API_KEY = api_key_cfg
TEMPERATURE = model_cfg.get("temperature", 0.2)

# 로컬 설정
QMLLINT_PATH   = config["tools"]["qmllint_path"]
TARGET_EXTS    = tuple(config["project"]["target_exts"])
APPLY_PATCH    = config["options"]["apply_patch"]

# Ollama 관련(선택): bin_dir / model_dir
OLLAMA_MODEL_DIR = config.get("ollama", {}).get("model_dir")
if OLLAMA_MODEL_DIR:
    if OLLAMA_MODEL_DIR.startswith("${") and OLLAMA_MODEL_DIR.endswith("}"):
        env_name = OLLAMA_MODEL_DIR[2:-1]
        OLLAMA_MODEL_DIR = os.getenv(env_name)
    OLLAMA_MODEL_DIR = Path(OLLAMA_MODEL_DIR).expanduser().resolve()

OLLAMA_BIN_DIR = config.get("ollama", {}).get("bin_dir")
if OLLAMA_BIN_DIR:
    if OLLAMA_BIN_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + OLLAMA_BIN_DIR

# ============================================================
# 3) 유틸리티
# ============================================================

def run_cmd(cmd: List[str], cwd: str = ".") -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def summarize_project(root: str = ".", max_chars: int = 12000) -> str:
    """
    프로젝트 내 TARGET_EXTS 파일만 읽어서 요약 텍스트 생성
    """
    buf, total = [], 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(TARGET_EXTS):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
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
    """
    unified diff 유효성 검증 후 git apply (repo 무관하게 시도 가능)
    """
    try:
        PatchSet(patch_text.splitlines(True))
    except Exception as e:
        return {"applied": False, "error": f"Invalid patch: {e}"}

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name

    code, out, err = run_cmd(["git", "apply", "--whitespace=fix", "--reject", tmp_path], cwd=repo_root)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return {"applied": code == 0, "stdout": out, "stderr": err, "code": code}

def git_commit_all(msg):
    run_cmd(["git", "add", "-A"])
    code, out, err = run_cmd(["git", "commit", "-m", msg])
    return (code == 0), out + err

def collect_qml_files(root: str) -> List[str]:
    qmls = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".qml"):
                qmls.append(os.path.join(dp, fn))
    return qmls

def run_qmllint(file_paths: List[str], module_paths: List[str] = None) -> str:
    """
    qmllint는 디렉터리 인자를 받지 않습니다. 반드시 QML 파일 리스트를 넘겨주세요.
    """
    if not file_paths:
        return "No QML files found."
    cmd = [QMLLINT_PATH]
    for mp in (module_paths or []):
        cmd += ["-I", mp]
    cmd += file_paths
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    text = out
    if err:
        text += "\nERR:\n" + err
    return text

def is_unified_diff(text: str) -> bool:
    if not text:
        return False
    return text.startswith("diff --git") or text.startswith("--- ") or text.startswith("Index: ")

def extract_assistant_text(run_res) -> str:
    """
    Autogen RunResponse / ChatResult / 기타 변형에서 안전하게 텍스트 추출.
    - .text / .content / .response 우선
    - messages(list)에서 마지막 assistant-like content를 수습
    - content가 list(parts)인 경우 합쳐서 반환
    """
    for attr in ("text", "content", "response"):
        val = getattr(run_res, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    msgs = getattr(run_res, "messages", None)
    if not msgs or not isinstance(msgs, list):
        return ""

    def parts_to_text(parts):
        if isinstance(parts, str):
            return parts
        if isinstance(parts, list):
            acc = []
            for p in parts:
                if isinstance(p, dict):
                    if "text" in p and isinstance(p["text"], str):
                        acc.append(p["text"])
                    elif "content" in p and isinstance(p["content"], str):
                        acc.append(p["content"])
                elif isinstance(p, str):
                    acc.append(p)
            return "\n".join(acc)
        return ""

    # 뒤에서부터 assistant-like 메시지 탐색
    for m in reversed(msgs):
        if isinstance(m, dict):
            role = m.get("role") or m.get("from") or m.get("sender")
            content = m.get("content") or m.get("text") or m.get("data")
            text = parts_to_text(content)
            if text.strip() and (role in ("assistant", "tool", "QMLCoder", "QMLStaticAnalyzer", "Reviewer") or role is None):
                return text.strip()
        else:
            role = getattr(m, "role", None) or getattr(m, "from", None) or getattr(m, "sender", None)
            content = getattr(m, "content", None) or getattr(m, "text", None)
            text = parts_to_text(content)
            if text.strip() and (role in ("assistant", "tool") or role is None):
                return text.strip()

    return ""

# ============================================================
# 4) LLM 에이전트 정의 (run() 기반) Autogen AssistantAgent 활용, 시스템 메시지로 역할/지침 부여
# ============================================================

llm_config = {
    "config_list": [
        {
            "model": model_cfg.get("model", MODEL_NAME),
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

# ============================================================
# 5) 메인 파이프라인
# ============================================================
import requests
import json

def test_raw_api():
    print("=== RAW API TEST ===")
    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role":"user", "content":"hi"}]
    }
    r = requests.post(url, headers=headers, json=data)
    print("status =", r.status_code)
    print("response =", r.text)

def test_run():
    print("=== TEST RUN ===")
    resp = code_agent.run(
        messages=[{"role": "user", "content": "hello"}]
    )
    print("API KEY =", OPENAI_API_KEY)
    print("TEST response =", resp)

    print("=== LLM CONFIG ===")
    print(llm_config)

def main():
    TASK_DESCRIPTION  = prompts["qmlcoder"]["task_description"]
    FITNESS_CRITERIA  = prompts["qmlcoder"]["fitness_criteria"]
    project_context   = summarize_project(".", max_chars=9000)

    test_run()  # 테스트 실행 (LLM 응답 메시지 구조 확인용)
    test_raw_api()  # 원시 API 테스트 (인증/연결 문제 확인용)
    
    # 0) (선택) 모델 준비: 이미 있으면 pull skip / 서버만 확인
    if OLLAMA_MODEL_DIR is not None:
        try:
            run_model(pull_model=MODEL_NAME, model_dir=OLLAMA_MODEL_DIR)
        except Exception as e:
            print(f"[WARN] run_model failed: {e}")

    # === 1단계: QML 코드 생성 ===
    gen_res = code_agent.run(
        f"{TASK_DESCRIPTION}\n\n"
        f"프로젝트 컨텍스트:\n{project_context}\n\n"
        "반드시 unified diff만 출력하세요. 설명/문장은 출력하지 마세요."
    )
    gen_patch = (extract_assistant_text(gen_res) or "").strip()

    print("\n=== CODE GENERATOR PATCH ===\n", gen_patch if gen_patch else "[EMPTY]")

    if not gen_patch or not is_unified_diff(gen_patch):
        print("[ERROR] QMLCoder output is empty or not a unified diff. Stopping pipeline.")
        # 디버깅용 원본 메시지 출력
        try:
            print("[DEBUG] gen_res.messages preview:", json.dumps(getattr(gen_res, "messages", []), ensure_ascii=False)[:1000])
        except Exception:
            pass
        return

    # === 2단계: 임시 디렉토리에 패치 적용 후 qmllint 실행 ===
    temp_dir = tempfile.mkdtemp(prefix="qml_agent_")
    try:
        # 프로젝트 파일 복사
        for name in os.listdir("."):
            if name.startswith(".git") or name == os.path.basename(temp_dir):
                continue
            src = os.path.join(".", name)
            dst = os.path.join(temp_dir, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        # 패치 적용
        if gen_patch.startswith(("diff --git", "--- ", "Index: ")):
            apply_res = apply_unified_diff(gen_patch, repo_root=temp_dir)
        else:
            apply_res = {"applied": False, "error": "No diff returned"}

        # qmllint 실행 (파일 리스트만)
        if apply_res.get("applied"):
            qml_files = collect_qml_files(temp_dir)
            qmllint_out = run_qmllint(qml_files)
        else:
            # 원본 대상으로라도 시도
            qml_files = collect_qml_files(".")
            qmllint_out = "패치 적용 실패 → 원본 대상으로 qmllint 실행\n" + run_qmllint(qml_files)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n=== QMLLINT RESULT ===\n", qmllint_out if qmllint_out else "[EMPTY]")

    # === 3단계: qmllint 기반 보완 패치 ===
    static_res = static_agent.run(
        "다음은 qmllint 결과입니다. 문제가 되는 부분을 수정하는 diff만 출력하세요.\n\n"
        f"{qmllint_out}\n"
    )
    static_patch = (extract_assistant_text(static_res) or "").strip()
    print("\n=== STATIC ANALYZER PATCH ===\n", static_patch if static_patch else "[EMPTY]")

    # === 4단계: Reviewer 최종 패치 ===
    review_res = review_agent.run(
        f"[FITNESS_CRITERIA]\n{FITNESS_CRITERIA}\n\n"
        f"[GEN_PATCH]\n{gen_patch}\n\n"
        f"[QMLLINT]\n{qmllint_out}\n\n"
        f"[STATIC_PATCH]\n{static_patch}\n"
        "최종 승인된 unified diff만 출력하세요."
    )
    final_patch = (extract_assistant_text(review_res) or "").strip()
    print("\n=== FINAL APPROVED PATCH ===\n", final_patch if final_patch else "[EMPTY]")

    # === 적용 옵션 ===
    if APPLY_PATCH and final_patch and is_unified_diff(final_patch):
        apply_res = apply_unified_diff(final_patch)
        if apply_res.get("applied"):
            ok, log = git_commit_all("agent: apply final QML patch")
            print("[INFO] 패치 적용 및 커밋 완료." if ok else f"[WARN] 커밋 실패: {log}")
        else:
            print("[ERROR] 최종 패치 적용 실패:", apply_res)

if __name__ == "__main__":
    main()
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
from openai import OpenAI        # 최신 ChatCompletion 엔진 사용
#from scripts.install_model import run_model


import html

def normalize_unified_diff(text: str) -> str:
    """백틱 코드펜스 제거 + HTML 엔티티 복원 + diff --git 자동 보정"""
    if not text:
        return ""

    # 1) ``` 코드펜스 줄만 제거하고 내부 내용은 유지
    cleaned_lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        cleaned_lines.append(ln)
    s = "\n".join(cleaned_lines)

    # 2) HTML 엔티티 역변환 (&lt; → <)
    s = html.unescape(s).strip()

    # 3) diff --git 자동 삽입
    final = []
    lns = s.splitlines()
    for i, ln in enumerate(lns):
        if ln.startswith("--- "):
            if i + 1 < len(lns) and lns[i+1].startswith("+++ "):
                a = ln.split()[1]
                b = lns[i+1].split()[1]
                final.append(f"diff --git {a} {b}")
        final.append(ln)

    return "\n".join(final).strip()

# ============================================================
# 1) 경로/설정 로딩
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS_PATH = SCRIPT_DIR / "prompts" / "prompts.yaml"
CONFIG_PATH  = SCRIPT_DIR / "config"  / "config.yaml"
MODELS_PATH  = SCRIPT_DIR / "config"  / "models.yaml"

prompts = load_yaml(PROMPTS_PATH)
config  = load_yaml(CONFIG_PATH)
models  = load_yaml(MODELS_PATH)

# ============================================================
# 2) 모델/경로 설정
# ============================================================

default_model_name = models["default_model"]
model_cfg = models["models"][default_model_name]

MODEL_NAME = model_cfg.get("model", default_model_name)       # 실제 서버 모델명 ex) llama3.1:8b
OPENAI_API_BASE = model_cfg["api_base"]
OPENAI_API_KEY  = model_cfg["api_key"]
TEMPERATURE = model_cfg.get("temperature", 0.2)

# OpenAI 호환 클라이언트(Ollama/Litellm/OpenAI 모두 동일)
client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

# ============================================================
# 3) 유틸
# ============================================================

def chat(messages, temperature=None):
    """OpenAI ChatCompletions 호출 (Ollama 지원)"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature if temperature is not None else TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()

def summarize_project(root=".", max_chars=12000):
    buf, total = [], 0
    exts = tuple(config["project"]["target_exts"])
    for dp,_,files in os.walk(root):
        for fn in files:
            if not fn.endswith(exts):
                continue
            path = os.path.join(dp, fn)
            try:
                txt = open(path, "r", encoding="utf-8").read()
            except:
                continue
            header = f"\n\n### FILE: {os.path.relpath(path, root)}\n"
            chunk = header + txt
            if total + len(chunk) > max_chars:
                buf.append(header + txt[:max_chars-total] + "\n...[TRUNCATED]")
                return "".join(buf)
            buf.append(chunk)
            total += len(chunk)
    return "".join(buf)

def run_cmd(cmd, cwd="."):
    p = subprocess.Popen(cmd, cwd=cwd,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def apply_unified_diff(diff_text, repo_root="."):
    """git apply"""
    try:
        PatchSet(diff_text.splitlines(True))
    except Exception as e:
        return {"applied": False, "error": str(e)}

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(diff_text)
        tmp_path = tmp.name
    code, out, err = run_cmd(["git","apply","--whitespace=fix","--reject", tmp_path],
                             cwd=repo_root)
    os.unlink(tmp_path)
    return {"applied": code==0, "stdout": out, "stderr": err}

def is_unified_diff(text):
    if not text:
        return False
    return text.startswith("diff --git") or text.startswith("--- ") or text.startswith("Index: ")

def collect_qml_files(root="."):
    out=[]
    for dp,_,files in os.walk(root):
        for fn in files:
            if fn.endswith(".qml"):
                out.append(os.path.join(dp, fn))
    return out

def run_qmllint(files):
    if not files:
        return "No QML files found."
    cmd = [config["tools"]["qmllint_path"]] + files
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    txt = out
    if err:
        txt += "\nERR:\n" + err
    return txt

# ============================================================
# 4) 실행 테스트
# ============================================================

def test_run():
    print("=== TEST RUN ===")
    msg = chat([{"role":"user","content":"hello"}], temperature=0.0)
    print("TEST output =", msg)
    print("=== LLM CONFIG ===")
    print({"model":MODEL_NAME,"base_url":OPENAI_API_BASE})

def test_raw_api():
    import requests
    print("=== RAW API TEST ===")
    r = requests.post(
        f"{OPENAI_API_BASE}/chat/completions",
        headers={"Authorization":f"Bearer {OPENAI_API_KEY}"},
        json={"model":MODEL_NAME,
              "messages":[{"role":"user","content":"hi"}]}
    )
    print("status=", r.status_code)
    print("response=", r.text[:300])

# ============================================================
# 5) 메인 파이프라인 (최신 방식)
# ============================================================

def main():

    # 0) 테스트
    test_run()
    test_raw_api()

    TASK = prompts["qmlcoder"]["task_description"]
    FIT  = prompts["qmlcoder"]["fitness_criteria"]
    ctx  = summarize_project(".", max_chars=9000)

    # === 1) QML 코드 생성 ===
    prompt_gen = (
        f"{TASK}\n\n"
        f"프로젝트 컨텍스트:\n{ctx}\n\n"
        "출력은 unified diff ONLY. 설명 금지."
    )

    gen_raw = chat([
        {"role": "system", "content": prompts["qmlcoder"]["system"]},
        {"role": "user", "content": prompt_gen}
    ])

    gen = normalize_unified_diff(gen_raw)

    print("=== CODE GENERATOR PATCH ===")
    print(gen if gen else "[EMPTY]")

    if not is_unified_diff(gen):
        print("[ERROR] generator didn't produce unified diff")
        print("[DEBUG raw]:", gen_raw)
        print("[DEBUG norm]:", gen)
        return

    # === 2) 패치 적용 + qmllint ===
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

        applied = apply_unified_diff(gen, repo_root=temp)
        if applied.get("applied"):
            qmls = collect_qml_files(temp)
            lint = run_qmllint(qmls)
        else:
            qmls = collect_qml_files(".")
            lint = run_qmllint(qmls)
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    print("\n=== QMLLINT ===\n", lint)

    # === 3) Static Fix ===
    static = chat([
        {"role":"system", "content":prompts["static_analyzer"]["system"]},
        {"role":"user",   "content":f"qmllint 출력입니다. 문제가 있는 부분만 고치는 diff를 출력:\n{lint}"}
    ])
    print("\n=== STATIC PATCH ===\n", static)

    # === 4) Reviewer ===
    review_prompt = (
        f"[FITNESS]\n{FIT}\n\n"
        f"[GEN]\n{gen}\n\n"
        f"[LINT]\n{lint}\n\n"
        f"[STATIC]\n{static}\n\n"
        "최종 승인된 unified diff만 출력."
    )
    final = chat([
        {"role":"system","content":prompts["reviewer"]["system"]},
        {"role":"user","content":review_prompt}
    ])

    print("\n=== FINAL PATCH ===\n", final if final else "[EMPTY]")

    if config["options"]["apply_patch"] and is_unified_diff(final):
        applied = apply_unified_diff(final)
        if applied.get("applied"):
            os.system('git add -A')
            os.system('git commit -m "agent: apply final QML patch"')
            print("[OK] applied+committed")
        else:
            print("[ERR] apply failed:", applied)

if __name__ == "__main__":
    main()
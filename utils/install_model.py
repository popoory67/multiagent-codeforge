# install_ollama_core.py
# -*- coding: utf-8 -*-

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen, Request

OLLAMA_API_URLS = [
    "http://127.0.0.1:11434/v1/models",
    "http://127.0.0.1:11434/api/tags",
]

def run(cmd, check=True, shell=False):
    print(f"[RUN] {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    return subprocess.run(cmd, check=check, shell=shell)


def which(cmd):
    return shutil.which(cmd) is not None


def ollama_in_path():
    return which("ollama")


def get_ollama_version():
    try:
        out = subprocess.check_output(["ollama", "--version"], text=True)
        return out.strip()
    except Exception:
        return ""

def ollama_installed():
    system = platform.system()

    if system == "Windows":
        default_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        return os.path.isfile(default_path)

    return shutil.which("ollama") is not None

# ----------------------------------------------------------
# Windows PATH 자동 보정
# ----------------------------------------------------------

def add_windows_ollama_to_path():
    if platform.system() != "Windows":
        return False

    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama")
    exe_path = os.path.join(candidate, "ollama.exe")

    if os.path.isfile(exe_path):
        path_val = os.environ.get("PATH", "")
        if candidate not in path_val:
            os.environ["PATH"] = path_val + (os.pathsep if path_val else "") + candidate
        print(f"[INFO] Windows PATH adjusted: {candidate}")
        return True
    return False


# ----------------------------------------------------------
# 서버 확인 / 자동 기동
# ----------------------------------------------------------

def wait_for_server(timeout_sec=20, sleep_sec=1):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for url in OLLAMA_API_URLS:
            try:
                req = Request(url, headers={"User-Agent": "curl/8.0"})
                with urlopen(req, timeout=2) as resp:
                    # HTTPResponse.status는 3.9+에서 사용 가능
                    status = getattr(resp, "status", None) or resp.getcode()
                    if status in (200, 401, 403):
                        print(f"[OK] Server response: {url} -> {status}")
                        return True
            except Exception:
                pass
        time.sleep(sleep_sec)
    return False


def start_server_best_effort():
    try:
        system = platform.system()
        if system == "Windows":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                preexec_fn=os.setpgrp,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as e:
        print(f"[WARN] Failed to start server: {e}")


# ----------------------------------------------------------
# OS별 설치
# ----------------------------------------------------------

def install_on_windows():
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "irm https://ollama.com/install.ps1 | iex",
    ]
    run(cmd)


def install_on_macos():
    if not which("brew"):
        raise RuntimeError("Homebrew가 필요합니다. https://brew.sh")
    run(["brew", "install", "ollama"])
    run(["brew", "services", "start", "ollama"], check=False)


def install_on_linux():
    if not which("curl"):
        raise RuntimeError("curl이 없습니다. 설치 후 다시 시도하세요.")
    run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)


# ----------------------------------------------------------
# 최종 install_ollama 함수 (IMPORT SAFE)
# ----------------------------------------------------------
def model_exists(model_name: str, model_dir: Path) -> bool:
    """
    모델 디렉토리를 기반으로 파일 존재 여부를 체크.
    Ollama는 'llama3.1:8b' → 'llama3.1_8b' 형태로 디렉토리 생성함.
    """
    safe_name = model_name.replace(":", "_")
    model_path = model_dir / safe_name
    return model_path.exists()

def run_model(pull_model=None, model_dir=None, check_only=False):

    if model_dir is None:
        raise ValueError("model_dir must be provided (from YAML).")

    # Ollama 설치 여부 확인 (설치 재시도 X)
    if not ollama_installed():
        print("[ERROR] Ollama not installed. Please install manually.")
        return False
    else:
        print("[INFO] Ollama installation detected.")

    print("[INFO] Checking server...")
    if not wait_for_server(5):
        print("[INFO] Server not running → starting...")
        start_server_best_effort()
        if not wait_for_server(15):
            print("[ERROR] Server still not responding.")
            return False

    # 모델 확인
    if pull_model:
        if model_exists(pull_model, model_dir):
            print(f"[INFO] Model already exists: {pull_model}")
        else:
            print(f"[INFO] Pulling model: {pull_model}")
            run(["ollama", "pull", pull_model])

    print("[DONE] install_model finished.")
    return True
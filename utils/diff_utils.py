# utils/diff_utils.py
import html
import tempfile
import os
import subprocess
from unidiff import PatchSet
from typing import Sequence, Tuple, Optional
from pathlib import Path

def run_cmd(cmd: Sequence[str], cwd: Optional[str | Path] = ".", timeout: Optional[int] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err

def normalize_unified_diff(text: str) -> str:
    if not text:
        return ""
    cleaned = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        cleaned.append(ln)
    s = html.unescape("\n".join(cleaned)).strip()

    final = []
    lns = s.splitlines()
    for i, ln in enumerate(lns):
        if ln.startswith("--- "):
            if i+1 < len(lns) and lns[i+1].startswith("+++ "):
                a = ln.split()[1]
                b = lns[i+1].split()[1]
                final.append(f"diff --git {a} {b}")
        final.append(ln)
    return "\n".join(final).strip()

def is_unified_diff(text):
    if not text:
        return False
    return text.startswith("diff --git") or text.startswith("--- ") or text.startswith("Index: ")

def apply_unified_diff(diff_text: str, repo_root="."):
    try:
        PatchSet(diff_text.splitlines(True))
    except Exception as e:
        return {"applied": False, "error": str(e)}

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(diff_text)
        tmp_path = tmp.name

    code, out, err = run_cmd(["git", "apply", "--whitespace=fix", "--reject", tmp_path], cwd=repo_root)
    os.unlink(tmp_path)
    return {"applied": code == 0, "stdout": out, "stderr": err}
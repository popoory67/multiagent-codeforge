# utils/project_utils.py
import os

def summarize_project(root, exts, max_chars=12000):
    buf, total = [], 0
    for dp, _, files in os.walk(root):
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
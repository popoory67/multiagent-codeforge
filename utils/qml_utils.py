# utils/qml_utils.py
import os
import subprocess

def collect_qml_files(root="."):
    out = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".qml"):
                out.append(os.path.join(dp, fn))
    return out

def run_qmllint(files, qmllint_path):
    if not files:
        return "No QML files found."
    cmd = [qmllint_path] + files
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    txt = out
    if err:
        txt += "\nERR:\n" + err
    return txt
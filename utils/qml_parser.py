# utils/qml_parser.py
"""Lightweight QML file parser — extracts functions, signals, properties, bindings, and child components."""

import re
from pathlib import Path


def parse_qml_file(filepath: str) -> dict:
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")

    functions = []
    signals = []
    properties = []
    bindings = []
    children = set()

    # Root component (first top-level Type { )
    root_match = re.match(r'\s*(?:import\s+[^\n]+\n)*\s*(\w+)\s*\{', text)
    root_type = root_match.group(1) if root_match else "Unknown"

    # function name(params) [: returnType]
    for m in re.finditer(r'function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?', text):
        functions.append({
            "name": m.group(1),
            "params": m.group(2).strip(),
            "return_type": m.group(3) or "",
        })

    # signal name[(params)]
    for m in re.finditer(r'signal\s+(\w+)\s*(?:\(([^)]*)\))?', text):
        signals.append({
            "name": m.group(1),
            "params": (m.group(2) or "").strip(),
        })

    # [readonly|default] property <type> <name>
    for m in re.finditer(r'(readonly\s+|default\s+)?property\s+(\w+)\s+(\w+)', text):
        properties.append({
            "qualifier": (m.group(1) or "").strip(),
            "type": m.group(2),
            "name": m.group(3),
        })

    # Property bindings: key: value (inside component body)
    # Match lines like:  name: "something"  /  screenId: "F_53"  /  priority: 1
    # Skip import lines, function bodies, and nested objects
    for m in re.finditer(
        r'^\s+(\w+)\s*:\s*(.+?)$', text, re.MULTILINE
    ):
        key = m.group(1)
        value = m.group(2).strip()
        # Skip known QML structural keywords and signal handlers
        if key in ('id', 'anchors', 'x', 'y', 'z', 'width', 'height',
                   'visible', 'enabled', 'opacity', 'clip', 'focus',
                   'Layout', 'left', 'right', 'top', 'bottom',
                   'horizontalCenter', 'verticalCenter', 'fill',
                   'margins', 'leftMargin', 'rightMargin',
                   'topMargin', 'bottomMargin', 'centerIn',
                   'source', 'color', 'radius', 'border'):
            continue
        # Skip signal handlers (onXxx)
        if key.startswith('on') and len(key) > 2 and key[2].isupper():
            continue
        bindings.append({
            "key": key,
            "value": value.rstrip(';'),
        })

    # Child components: CapitalizedWord {  (excluding root)
    for m in re.finditer(r'^\s+([A-Z]\w+)\s*\{', text, re.MULTILINE):
        children.add(m.group(1))

    return {
        "file": filepath,
        "root_type": root_type,
        "functions": functions,
        "signals": signals,
        "properties": properties,
        "bindings": bindings,
        "children": sorted(children),
    }

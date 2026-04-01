"""
This script converts all .docx and .pdf files in a specified folder to .md format.

Usage:
    python run_convert_to_md.py <input_folder> [-o <output_folder>]

if -o/--output is not specified, the converted .md files will be saved in the same location as the source files.
"""
import sys
import subprocess
import argparse
from pathlib import Path

# Ensure required packages are installed before importing them
#   pip install python-docx pymupdf
#   pip install --upgrade python-docx pymupdf
def _ensure_packages():
    missing = []
    try:
        import docx  # noqa: F401
    except ImportError:
        missing.append("python-docx")
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        missing.append("pymupdf")
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_packages()

from utils.docx_reader import docx_to_md  # noqa: E402
import pymupdf  # noqa: E402


def pdf_to_md(pdf_path: Path) -> str:
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n---\n\n".join(pages)


def convert_file(src: Path, out_dir: Path | None) -> Path:
    ext = src.suffix.lower()
    if ext == ".docx":
        md = docx_to_md(src)
    elif ext == ".pdf":
        md = pdf_to_md(src)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / src.with_suffix(".md").name
    else:
        dest = src.with_suffix(".md")

    dest.write_text(md, encoding="utf-8")
    return dest


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Convert docx/pdf files in a folder to md format")
    parser.add_argument("input_folder", help="Path to the folder containing files to convert")
    parser.add_argument("-o", "--output", help="Output folder (if not specified, files will be saved in the same location)")
    args = parser.parse_args()

    src_dir = Path(args.input_folder)
    if not src_dir.is_dir():
        print(f"Folder not found: {src_dir}")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else None

    files = sorted(
        f for f in src_dir.rglob("*")
        if f.suffix.lower() in (".docx", ".pdf") and not f.name.startswith("~")
    )

    if not files:
        print("No docx/pdf files to convert.")
        sys.exit(0)

    print(f"Total {len(files)} files conversion started\n")

    for f in files:
        try:
            dest = convert_file(f, out_dir)
            print(f"  ✓ {f.name} → {dest}")
        except Exception as e:
            print(f"  ✗ {f.name} — {e}")

    print("\nConversion completed.")


if __name__ == "__main__":
    main()

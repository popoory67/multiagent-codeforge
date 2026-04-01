# utils/docx_reader.py
import sys
import argparse
from pathlib import Path
from docx import Document


# Heading style name -> markdown level
HEADING_MAP = {
    "Heading 1": 1, "Heading 2": 2, "Heading 3": 3,
    "Heading 4": 4, "Heading 5": 5, "Heading 6": 6,
}


def _heading_level(paragraph):
    style = paragraph.style
    if style and style.name in HEADING_MAP:
        return HEADING_MAP[style.name]
    # outline_level from XML (some docs use outline instead of named styles)
    try:
        if style and style.paragraph_format:
            lvl = style.paragraph_format.outline_level
            if lvl is not None and 0 <= lvl <= 5:
                return lvl + 1
    except AttributeError:
        pass
    return None


def _is_list_item(paragraph):
    """Check if paragraph has numbering (bullet / numbered list)."""
    pPr = paragraph._element.find(
        './/{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numPr'
    )
    return pPr is not None


def _table_to_md(table):
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    separator = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
    return rows[0] + "\n" + separator + "\n" + "\n".join(rows[1:])


def docx_to_md(docx_path):
    doc = Document(docx_path)
    lines = []

    # Build a set of table element refs so we can insert them inline
    table_elements = {tbl._element: tbl for tbl in doc.tables}
    table_inserted = set()

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]

        if tag == "p":
            # Find the matching paragraph object
            para = None
            for p in doc.paragraphs:
                if p._element is block:
                    para = p
                    break
            if para is None:
                continue

            text = para.text.strip()
            if not text:
                continue

            level = _heading_level(para)
            if level:
                lines.append(f"\n{'#' * level} {text}\n")
            elif _is_list_item(para):
                lines.append(f"- {text}")
            else:
                lines.append(text)

        elif tag == "tbl":
            if block in table_elements and id(block) not in table_inserted:
                table_inserted.add(id(block))
                lines.append("")
                lines.append(_table_to_md(table_elements[block]))
                lines.append("")

    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Convert .docx to markdown")
    parser.add_argument("input", help="Path to .docx file")
    parser.add_argument("-o", "--output", help="Output path (default: same name .md)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".md")

    md = docx_to_md(input_path)
    output_path.write_text(md, encoding="utf-8")
    print(f"Saved: {output_path} ({len(md)} chars)")


if __name__ == "__main__":
    main()

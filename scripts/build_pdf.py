"""Render a Markdown document to a typeset PDF.

Built because this machine has no pandoc and no LaTeX. Handles the subset of Markdown the Q-Neuro
documents actually use: headings, paragraphs with inline emphasis and code, bullet and numbered
lists, fenced code blocks, block quotes, pipe tables, images and horizontal rules.

    python scripts/build_pdf.py docs/PAPER_STUDENT.md paper/qneuro-paper.pdf --title "..."
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]

INK = colors.HexColor("#141D1B")
MUTED = colors.HexColor("#5B6764")
ACCENT = colors.HexColor("#1B6D60")
FAIL = colors.HexColor("#A03726")
RULE = colors.HexColor("#DCDFD8")
SOFT = colors.HexColor("#F2F4F1")

PAGE_WIDTH = A4[0] - 40 * mm

#: Helvetica and Courier are base-14 fonts limited to Latin-1, so every Greek letter and maths
#: operator in these documents renders as a hollow box. DejaVu covers all of them and ships inside
#: matplotlib, which is already a dependency, so the build stays reproducible from the lockfile.
FALLBACK = "DejaVuSans"
FALLBACK_MONO = "DejaVuSansMono"


def register_fallback_fonts() -> None:
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    source = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for family, suffixes in (
        (FALLBACK, ("", "-Bold", "-Oblique", "-BoldOblique")),
        (FALLBACK_MONO, ("", "-Bold", "-Oblique", "-BoldOblique")),
    ):
        faces = []
        for suffix in suffixes:
            name = f"{family}{suffix}"
            pdfmetrics.registerFont(TTFont(name, str(source / f"{name}.ttf")))
            faces.append(name)
        pdfmetrics.registerFontFamily(family, normal=faces[0], bold=faces[1],
                                      italic=faces[2], boldItalic=faces[3])


def _renderable(character: str) -> bool:
    try:
        character.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


def unicode_safe(text: str, family: str = FALLBACK) -> str:
    """Wrap every run of non-Latin-1 characters in the fallback face.

    Runs are wrapped rather than individual characters so a formula keeps its kerning. `<font face>`
    overrides the weight reportlab would otherwise infer, so the surrounding `<b>`/`<i>` state is
    tracked and the matching DejaVu face named explicitly -- otherwise a Greek letter inside a bold
    sentence comes out at regular weight.
    """

    suffix = {(False, False): "", (True, False): "-Bold",
              (False, True): "-Oblique", (True, True): "-BoldOblique"}
    out: list[str] = []
    run: list[str] = []
    bold = italic = False
    index = 0

    def flush() -> None:
        if run:
            face = family + suffix[(bold, italic)]
            out.append(f'<font face="{face}">{"".join(run)}</font>')
            run.clear()

    while index < len(text):
        character = text[index]
        if character == "<":
            end = text.find(">", index)
            if end != -1:
                flush()
                tag = text[index:end + 1]
                lowered = tag.lower()
                if lowered in ("<b>", "<strong>"):
                    bold = True
                elif lowered in ("</b>", "</strong>"):
                    bold = False
                elif lowered in ("<i>", "<em>"):
                    italic = True
                elif lowered in ("</i>", "</em>"):
                    italic = False
                out.append(tag)
                index = end + 1
                continue
        if _renderable(character):
            flush()
            out.append(character)
        else:
            run.append(character)
        index += 1
    flush()
    return "".join(out)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    common = {"fontName": "Helvetica", "textColor": INK, "leading": 14.5}
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=26, leading=31, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13, leading=18,
                                   textColor=MUTED, alignment=TA_LEFT, spaceAfter=18),
        "byline": ParagraphStyle("byline", fontName="Helvetica", fontSize=10, leading=15,
                                 textColor=MUTED, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=18, leading=23,
                             textColor=INK, spaceBefore=20, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13.5, leading=18,
                             textColor=INK, spaceBefore=15, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11, leading=15,
                             textColor=ACCENT, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("body", fontSize=9.7, spaceAfter=7, **common),
        "quote": ParagraphStyle("quote", fontSize=10, leading=15, textColor=INK,
                                fontName="Helvetica-Oblique", leftIndent=12, rightIndent=8,
                                spaceBefore=6, spaceAfter=8, borderPadding=6,
                                backColor=SOFT),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.2, leading=10.6,
                               textColor=INK, backColor=SOFT, borderPadding=6,
                               spaceBefore=5, spaceAfter=8),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.3,
                                  leading=11, textColor=MUTED, alignment=TA_CENTER,
                                  spaceBefore=3, spaceAfter=12),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.2, leading=10.5,
                               textColor=INK),
        "cellhead": ParagraphStyle("cellhead", fontName="Helvetica-Bold", fontSize=8.2,
                                   leading=10.5, textColor=colors.white),
    }


def inline(text: str) -> str:
    """Markdown inline emphasis to reportlab markup, escaping everything else."""

    placeholders: list[str] = []

    def stash(match: re.Match, template: str) -> str:
        body = unicode_safe(html.escape(match.group(1)), FALLBACK_MONO)
        placeholders.append(template.format(body))
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: stash(m, '<font face="Courier" size="8.6">{}</font>'), text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = unicode_safe(text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def build_table(rows: list[str], style: dict):
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed.append(cells)
    if len(parsed) >= 2 and set(parsed[1][0]) <= set("-: "):
        header, body = parsed[0], parsed[2:]
    else:
        header, body = parsed[0], parsed[1:]
    columns = len(header)
    data = [[Paragraph(inline(c), style["cellhead"]) for c in header]]
    for row in body:
        row = (row + [""] * columns)[:columns]
        data.append([Paragraph(inline(c), style["cell"]) for c in row])
    width = PAGE_WIDTH / columns
    table = Table(data, colWidths=[width] * columns, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def image_flowable(path: Path, caption: str, style: dict) -> list:
    if not path.is_file():
        return [Paragraph(f"[missing figure: {path.name}]", style["caption"])]
    from PIL import Image as PILImage

    with PILImage.open(path) as handle:
        width, height = handle.size
    scale = min(PAGE_WIDTH / width, 1.0)
    flowables = [Spacer(1, 4), Image(str(path), width=width * scale, height=height * scale)]
    if caption:
        flowables.append(Paragraph(inline(caption), style["caption"]))
    else:
        flowables.append(Spacer(1, 10))
    return flowables


def convert(markdown: str, style: dict) -> list:
    story: list = []
    lines = markdown.split("\n")
    index = 0
    pending_bullets: list[str] = []
    pending_numbers: list[str] = []

    def flush_lists():
        nonlocal pending_bullets, pending_numbers
        for items, kind in ((pending_bullets, "bullet"), (pending_numbers, "1")):
            if items:
                story.append(ListFlowable(
                    [ListItem(Paragraph(inline(i), style["body"]), leftIndent=14)
                     for i in items],
                    bulletType=kind, start="1" if kind == "1" else None,
                    leftIndent=14, bulletFontSize=8,
                ))
                story.append(Spacer(1, 4))
        pending_bullets, pending_numbers = [], []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_lists()
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            text = html.escape("\n".join(block)).replace(" ", "&nbsp;").replace("\n", "<br/>")
            text = unicode_safe(text, FALLBACK_MONO)
            story.append(Paragraph(text, style["code"]))
            continue

        if stripped.startswith("!["):
            flush_lists()
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                target = (ROOT / match.group(2).replace("../", "")).resolve()
                if not target.is_file():
                    target = (ROOT / "research" / "figures" / "generated" /
                              Path(match.group(2)).name)
                story.extend(image_flowable(target, match.group(1), style))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and "|" in lines[index + 1]:
            flush_lists()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index])
                index += 1
            story.append(build_table(rows, style))
            story.append(Spacer(1, 10))
            continue

        if stripped in ("---", "***", "___"):
            flush_lists()
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", color=RULE, thickness=0.8))
            story.append(Spacer(1, 6))
            index += 1
            continue

        if stripped.startswith("> "):
            flush_lists()
            block = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                block.append(lines[index].strip().lstrip(">").strip())
                index += 1
            story.append(Paragraph(inline(" ".join(b for b in block if b)), style["quote"]))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_lists()
            level = len(heading.group(1))
            key = "h1" if level <= 1 else ("h2" if level == 2 else "h3")
            if level == 1 and story:
                story.append(PageBreak())
            story.append(Paragraph(inline(heading.group(2)), style[key]))
            index += 1
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            pending_numbers and flush_lists()
            pending_bullets.append(bullet.group(1))
            index += 1
            continue

        number = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if number:
            pending_bullets and flush_lists()
            pending_numbers.append(number.group(1))
            index += 1
            continue

        if not stripped:
            flush_lists()
            index += 1
            continue

        flush_lists()
        block = []
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^(#{1,6}\s|[-*]\s|\d+[.)]\s|\||>|```|!\[|---$)", lines[index].strip()
        ):
            block.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline(" ".join(block)), style["body"]))

    flush_lists()
    return story


def decorate(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, document.title_text)
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(canvas.getPageNumber()))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canvas.restoreState()


def build(source: Path, target: Path, title: str, subtitle: str, byline: str) -> None:
    register_fallback_fonts()
    style = styles()
    text = source.read_text(encoding="utf-8")
    # The first heading becomes the cover; the rest is the body.
    text = re.sub(r"\A#\s+.*?\n", "", text, count=1)
    text = re.sub(r"\A###?\s+.*?\n", "", text, count=1)

    story: list = [
        Spacer(1, 40 * mm),
        Paragraph(inline(title), style["title"]),
        Paragraph(inline(subtitle), style["subtitle"]),
        HRFlowable(width="35%", color=ACCENT, thickness=2, hAlign="LEFT"),
        Spacer(1, 10),
        Paragraph(byline, style["byline"]),
        PageBreak(),
    ]
    story.extend(convert(text, style))

    target.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
        title=title, author="Samyar Shafiee",
    )
    document.title_text = title
    document.build(story, onFirstPage=lambda c, d: None, onLaterPages=decorate)
    size = target.stat().st_size / 1024
    print(f"wrote {target}  ({size:.0f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--byline", default="Samyar Shafiee")
    args = parser.parse_args()
    build(args.source, args.target, args.title, args.subtitle, args.byline)


if __name__ == "__main__":
    main()

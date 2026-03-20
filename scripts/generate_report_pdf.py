from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report_pdf.md"
OUTPUT = ROOT / "report.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def build_pdf() -> None:
    pdfmetrics.registerFont(TTFont("ArialCyr", str(FONT_PATH)))

    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "Base",
        parent=styles["Normal"],
        fontName="ArialCyr",
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base,
        fontSize=18,
        leading=22,
        spaceBefore=8,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base,
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=8,
    )

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            story.append(Spacer(1, 0.15 * cm))
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:].strip()), h1))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(escape(line[3:].strip()), h2))
            index += 1
            continue
        if line.startswith("!["):
            start = line.find("](")
            end = line.rfind(")")
            path = line[start + 2 : end]
            image_path = (ROOT / path).resolve()
            if image_path.exists():
                story.append(Image(str(image_path), width=16.5 * cm, height=9.5 * cm))
                story.append(Spacer(1, 0.2 * cm))
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|---"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                raw = [escape(cell.strip()) for cell in lines[index].strip("|").split("|")]
                if set("".join(raw).replace(":", "").replace("-", "").strip()) == set():
                    index += 1
                    continue
                rows.append(raw)
                index += 1
            table = Table(rows, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), "ArialCyr"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEADING", (0, 0), (-1, -1), 11),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.2 * cm))
            continue
        if line.startswith("- "):
            story.append(Paragraph(escape("- " + line[2:].strip()), base))
            index += 1
            continue
        story.append(Paragraph(escape(line), base))
        index += 1

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf()

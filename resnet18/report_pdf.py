"""Render the completed Role 6 Markdown report as a clean group handover PDF."""
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image


def build_pdf(markdown_path, destination, comparison_figure=None):
    regular, bold = "Helvetica", "Helvetica-Bold"
    ink, accent = colors.HexColor("#173441"), colors.HexColor("#146C7A")
    body = ParagraphStyle("Body", fontName=regular, fontSize=9.4, leading=13.6,
                          textColor=ink, spaceAfter=8, allowWidows=0, allowOrphans=0)
    title = ParagraphStyle("Title", parent=body, fontName=bold, fontSize=23, leading=28,
                           spaceBefore=7, spaceAfter=16, keepWithNext=True)
    heading = ParagraphStyle("Heading", parent=body, fontName=bold, fontSize=12.5,
                             leading=16, spaceBefore=12, spaceAfter=7, textColor=accent,
                             keepWithNext=True)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.1, leading=10.8, spaceAfter=0)
    header = ParagraphStyle("TableHeader", parent=cell, fontName=bold, textColor=colors.white)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8.1, leading=11, textColor=colors.HexColor("#526773"))
    width, height = A4
    margin = 46
    usable = width - 2 * margin
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=margin,
                            leftMargin=margin, topMargin=49, bottomMargin=43,
                            title="CSCI435 Person 6 - Fine-tuned ResNet-18",
                            author="CSCI435 - Person 6")

    def inline(text):
        text = text.replace("—", "-").replace("–", "-").replace("‑", "-")
        text = escape(text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                      r'<link href="\2" color="#146C7A"><u>\1</u></link>', text)
        return re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.3">\1</font>', text)

    def page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4E0E5"))
        canvas.line(margin, height - 31, width - margin, height - 31)
        canvas.setFillColor(accent)
        canvas.setFont(bold, 8)
        canvas.drawString(margin, height - 23, "CSCI435  |  PERSON 6")
        canvas.setFillColor(colors.HexColor("#647985"))
        canvas.setFont(regular, 7.5)
        canvas.drawString(margin, 26, "Fine-tuned ResNet-18 experiment handover")
        canvas.drawRightString(width - margin, 26, str(document.page))
        canvas.restoreState()

    lines = Path(markdown_path).read_text(encoding="utf-8").splitlines()
    story, index = [], 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), title))
        elif line.startswith("## "):
            if line == "## Interpretation" and comparison_figure:
                story.append(Spacer(1, 5))
                story.append(Image(str(comparison_figure), width=usable, height=usable * 3.6 / 8))
                story.append(Paragraph("Combined test macro-F1 by repeated holdout. Bars show the exact frozen prerequisites and their selected fine-tuned models.", caption))
            story.append(Paragraph(inline(line[3:]), heading))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                parts = [p.strip() for p in lines[index].strip().split("|")[1:-1]]
                if not all(re.fullmatch(r"[-: ]+", p) for p in parts):
                    rows.append(parts)
                index += 1
            n = len(rows[0])
            fractions = [.12, .12, .19, .19, .18, .20] if n == 6 else [.22, .39, .39]
            table = Table([[Paragraph(inline(v), header if r == 0 else cell) for v in row]
                           for r, row in enumerate(rows)], colWidths=[usable * f for f in fractions],
                          repeatRows=1, hAlign=TA_LEFT)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F5F7"), colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#D4E0E5")),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        else:
            story.append(Paragraph(inline(line), body))
        index += 1
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return destination

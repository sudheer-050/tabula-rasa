"""Build paper.pdf from PAPER.md and the released figures; requires ReportLab."""
from pathlib import Path
import html
import re

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak,
)

ROOT = Path(__file__).resolve().parent
WIDTH = 468
INK = colors.HexColor("#182b42")
FONT = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
pdfmetrics.registerFont(TTFont("Equation", str(FONT)))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("PaperTitle", fontName="Times-Bold", fontSize=23, leading=26,
                          textColor=INK, spaceAfter=9))
styles.add(ParagraphStyle("PaperSubtitle", fontName="Times-Roman", fontSize=14, leading=18,
                          textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle("PaperBody", fontName="Times-Roman", fontSize=10.5, leading=14,
                          alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle("PaperH1", fontName="Times-Bold", fontSize=14, leading=17,
                          textColor=INK, spaceBefore=14, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle("PaperH2", fontName="Times-Bold", fontSize=11.5, leading=14,
                          textColor=INK, spaceBefore=10, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle("PaperCaption", fontName="Times-Roman", fontSize=9, leading=12,
                          spaceBefore=4, spaceAfter=10))
styles.add(ParagraphStyle("PaperCell", fontName="Times-Roman", fontSize=9, leading=11))
styles.add(ParagraphStyle("PaperRef", fontName="Times-Roman", fontSize=9.2, leading=12,
                          spaceAfter=6))


def inline(text):
    text = text.replace("\u2014", " - ").replace("\u2013", "-").replace("\u2011", "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<link href="{m[2]}" color="#175b91">{m[1]}</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.6">\1</font>', text)
    return text


def picture(path, max_height=445):
    w, h = ImageReader(str(path)).getSize()
    scale = min(WIDTH / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d6dee8"))
    canvas.setLineWidth(.5)
    canvas.line(72, 750, 540, 750)
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(72, 760, "ARC-AGI-3  |  EMPIRICAL ENGINEERING STUDY")
    canvas.drawRightString(540, 760, "GANDIBOYANA  |  OCTOBER 2026")
    canvas.setFillColor(colors.HexColor("#526277"))
    canvas.drawString(72, 33, "Useful Interaction Under a Fixed Compute Budget")
    canvas.drawRightString(540, 33, str(doc.page))
    canvas.restoreState()


def main():
    lines = (ROOT / "PAPER.md").read_text(encoding="utf8").splitlines()
    story, i, equation, references = [], 0, 0, False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("[PDF paper]"):
            i += 1
            continue
        if line == "The report makes four contributions:":
            items = [Paragraph(inline(line), styles["PaperBody"])]
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                items.append(Paragraph(inline(lines[i]), styles["PaperBody"]))
                i += 1
            story.append(KeepTogether(items))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["PaperTitle"]))
        elif line.startswith("## An Empirical"):
            story.append(Paragraph(inline(line[3:]), styles["PaperSubtitle"]))
        elif line.startswith("## "):
            references = line == "## References"
            if line.startswith(("## Appendix", "## 5. Results")):
                story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]), styles["PaperH1"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["PaperH2"]))
        elif line.startswith("$$"):
            equation += 1
            story.append(KeepTogether([Spacer(1, 3),
                         picture(ROOT / f"paper_equation_{equation}.png", 44), Spacer(1, 8)]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in cells):
                    rows.append([Paragraph(inline(x), styles["PaperCell"]) for x in cells])
                i += 1
            columns = len(rows[0])
            widths = [WIDTH * .42, WIDTH * .58] if columns == 2 else [WIDTH * .52, WIDTH * .24, WIDTH * .24]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#879db6")),
                ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#dce3eb")),
            ]))
            items = [Spacer(1, 4), table, Spacer(1, 3)]
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**Table"):
                items.append(Paragraph(inline(lines[j]), styles["PaperCaption"]))
                i = j + 1
            story.append(KeepTogether(items))
            continue
        elif line.startswith("!["):
            path = re.search(r"\]\(([^)]+)\)", line).group(1)
            items = [Spacer(1, 8), picture(ROOT / path)]
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**Figure"):
                items.append(Paragraph(inline(lines[j]), styles["PaperCaption"]))
                i = j
            story.append(KeepTogether(items))
        else:
            if re.match(r"\d+\. ", line) and not references:
                items = []
                while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                    items.append(Paragraph(inline(lines[i]), styles["PaperBody"]))
                    i += 1
                story.append(KeepTogether(items))
                continue
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "|", "![", "$$")):
                if re.match(r"\d+\. ", lines[i + 1]):
                    break
                i += 1
                paragraph.append(lines[i].strip())
            style = styles["PaperRef"] if references else styles["PaperBody"]
            story.append(Paragraph(inline(" ".join(paragraph)), style))
        i += 1
    doc = SimpleDocTemplate(str(ROOT / "paper.pdf"), pagesize=(612, 792),
                            leftMargin=72, rightMargin=72, topMargin=56, bottomMargin=56,
                            title="Useful Interaction Under a Fixed Compute Budget",
                            author="Sudheer Kumar Gandiboyana", subject="ARC-AGI-3 empirical engineering study")
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(f"Created {ROOT / 'paper.pdf'}")


if __name__ == "__main__":
    main()

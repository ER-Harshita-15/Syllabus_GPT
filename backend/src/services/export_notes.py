import os
from markdown import markdown
from bs4 import BeautifulSoup

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


EXPORT_DIR = "./exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


# ---------------------------------------------------
# CLEAN WORKING Markdown → Paragraphs conversion
# ---------------------------------------------------
def markdown_to_paragraphs(md_text: str):
    # Convert markdown → raw HTML
    html = markdown(md_text)

    # Parse clean HTML using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    paragraphs = []

    # Only process supported tags
    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = element.get_text().strip()

        if not text:
            continue

        # Heading 1
        if element.name == "h1":
            paragraphs.append(Paragraph(f"<b><font size=18>{text}</font></b>", normal))
            paragraphs.append(Spacer(1, 0.20 * inch))

        # Heading 2
        elif element.name == "h2":
            paragraphs.append(Paragraph(f"<b><font size=16>{text}</font></b>", normal))
            paragraphs.append(Spacer(1, 0.15 * inch))

        # Heading 3
        elif element.name == "h3":
            paragraphs.append(Paragraph(f"<b><font size=14>{text}</font></b>", normal))
            paragraphs.append(Spacer(1, 0.10 * inch))

        # Bullet list
        elif element.name == "li":
            paragraphs.append(Paragraph(f"• {text}", normal))
            paragraphs.append(Spacer(1, 0.10 * inch))

        # Normal paragraph
        elif element.name == "p":
            paragraphs.append(Paragraph(text, normal))
            paragraphs.append(Spacer(1, 0.12 * inch))

    return paragraphs


# ---------------------------------------------------
# MAIN — Generate Beautiful PDF
# ---------------------------------------------------
def generate_beautiful_pdf(markdown_text: str, filename: str, title: str, subject: str):
    pdf_path = os.path.join(EXPORT_DIR, filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # ---------------- Title Page ----------------
    title_style = ParagraphStyle(
        name="TitlePage",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=30,
        spaceAfter=30,
    )

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=16,
        spaceAfter=50,
    )

    elements.append(Paragraph(title, title_style))

    if subject:
        elements.append(Paragraph(f"<b>Subject:</b> {subject}", subtitle_style))

    # Page break before content
    elements.append(PageBreak())

    # --------------- Markdown → PDF content ---------------
    paragraphs = markdown_to_paragraphs(markdown_text)
    elements.extend(paragraphs)

    # Build final PDF
    doc.build(elements)

    return pdf_path

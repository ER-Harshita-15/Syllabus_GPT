from pdfminer.high_level import extract_text
import io



def extract_text_from_pdf(content):
    with io.BytesIO(content) as pdf_file:
        text = extract_text(pdf_file)
        return text

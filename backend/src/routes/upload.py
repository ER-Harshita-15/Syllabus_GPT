from fastapi import APIRouter, UploadFile, File
from src.services.pdf_extract import extract_text_from_pdf
from src.services.ocr import extract_text_from_image

router = APIRouter()

@router.post("/upload")
async def upload_syllabus(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()

    # PDF
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
        return {"status": "success", "text": text}

    # Image
    if filename.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(content)
        return {"status": "success", "text": text}

    # Plain text file
    text = content.decode("utf-8")
    return {"status": "success", "text": text}


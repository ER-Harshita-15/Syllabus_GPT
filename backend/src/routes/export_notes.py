from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.services.export_notes import generate_beautiful_pdf


router = APIRouter(
    prefix="/notes",
    tags=["Notes Export"]
)


class ExportPdfRequest(BaseModel):
    notes_markdown: str
    filename: str | None = "notes.pdf"
    title: str | None = "Syllabus GPT Notes"
    subject: str | None = ""
    

@router.post("/export/pdf")
def export_notes_pdf(req: ExportPdfRequest):
    """
    Convert Markdown notes → Beautiful PDF
    """
    try:
        pdf_path = generate_beautiful_pdf(
            markdown_text=req.notes_markdown,
            filename=req.filename or "notes.pdf",
            title=req.title or "Syllabus GPT Notes",
            subject=req.subject or "",
        )

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=req.filename or "notes.pdf"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )

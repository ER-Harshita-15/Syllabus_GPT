"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.notes_llm import generate_final_notes

router = APIRouter(
    prefix="/notes",
    tags=["Notes Generation"]
)

class NotesRequest(BaseModel):
    syllabus_text: str
    subject: str | None = None
    use_pyq: bool = False
    top_k: int = 10


@router.post("/generate")
def generate_notes(req: NotesRequest):

    #Generates final notes using HYDE + RAG + LLM
    try:
        final_notes = generate_final_notes(
            syllabus_text=req.syllabus_text,
            subject=req.subject,
            use_pyq=req.use_pyq,
            top_k=req.top_k,
        )

        return {"notes": final_notes}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Notes generation failed: {str(e)}"
        )
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.services.notes_llm import generate_final_notes
from src.services.export_notes import generate_beautiful_pdf
from src.services.vector_store import retrieve_relevant_context

router = APIRouter(
    prefix="/notes",
    tags=["Notes Generation"]
)


class NotesRequest(BaseModel):
    syllabus_text: str
    subject: str | None = None
    use_pyq: bool = False
    top_k: int = 10


class NotesAndPdfRequest(NotesRequest):
    filename: str | None = "notes.pdf"
    title: str | None = None   # optional custom title for PDF cover


@router.post("/generate")
def generate_notes(req: NotesRequest):
    """
    Step 1: Generate ONLY markdown notes (no PDF).
    Useful for preview or debugging.
    """
    try:
        # (A) Get RAG context
        context = retrieve_relevant_context(
            syllabus_text=req.syllabus_text,
            subject=req.subject,
            use_pyq=req.use_pyq,
            top_k=req.top_k,
        )

        # (B) Generate final notes markdown using LLM
        notes_md = generate_final_notes(
            syllabus_text=req.syllabus_text,
            subject=req.subject,
            use_pyq=req.use_pyq,
            top_k=req.top_k,
        )

        return {
            "context_length": len(context),
            "notes_markdown": notes_md,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Notes generation failed: {str(e)}"
        )


@router.post("/generate-and-export/pdf")
def generate_notes_and_pdf(req: NotesAndPdfRequest):
    """
    Full pipeline in ONE call:
      syllabus_text (+subject) -> RAG -> notes markdown -> PDF file
    """
    try:
        # (A) Generate full notes markdown
        notes_md = generate_final_notes(
            syllabus_text=req.syllabus_text,
            subject=req.subject,
            use_pyq=req.use_pyq,
            top_k=req.top_k,
        )

        if not notes_md or not notes_md.strip():
            raise RuntimeError("Generated notes are empty; cannot create PDF.")

        # (B) Decide title
        if req.title:
            title = req.title
        else:
            if req.subject:
                title = f"{req.subject} - Generated Notes"
            else:
                title = "Syllabus GPT - Generated Notes"

        filename = req.filename or "notes.pdf"

        # (C) Generate PDF from markdown
        pdf_path = generate_beautiful_pdf(
            markdown_text=notes_md,
            filename=filename,
            title=title,
            subject=req.subject or "",
        )

        # (D) Return file
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=filename
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Notes generation failed: {str(e)}"
        )

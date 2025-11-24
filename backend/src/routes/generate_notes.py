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
    """
    Generates final notes using HYDE + RAG + LLM
    """
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

from fastapi import APIRouter
from pydantic import BaseModel

from src.services.vector_store import vector_search, retrieve_relevant_context

router = APIRouter(
    prefix="/retrieve",
    tags=["Retrieve / RAG"]
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ContextRequest(BaseModel):
    syllabus_text: str
    subject: str = None
    use_pyq: bool = False
    top_k: int = 10


@router.post("/query")
def raw_query(req: QueryRequest):
    return vector_search(req.query, req.top_k)


@router.post("/context")
def get_context(req: ContextRequest):
    ctx = retrieve_relevant_context(
        syllabus_text=req.syllabus_text,
        subject=req.subject,
        use_pyq=req.use_pyq,
        top_k=req.top_k
    )
    return {"context": ctx}

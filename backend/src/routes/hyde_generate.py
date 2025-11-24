from fastapi import APIRouter
from pydantic import BaseModel
from src.services.hyde_llm import generate_hyde_document

router = APIRouter(
    prefix="/hyde",
    tags=["HyDE Generator"]
)

class Topic(BaseModel):
    topic: str

@router.post("/generate")
def hyde_generate(data: Topic):
    hyde_doc = generate_hyde_document(data.topic)
    return {"hyde_doc": hyde_doc}

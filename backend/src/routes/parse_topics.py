from fastapi import APIRouter
from pydantic import BaseModel
from src.services.hyde_llm import parse_syllabus_into_topics

router = APIRouter()

class SyllabusData(BaseModel):
    text: str

@router.post("/parse-topics")
def parse_topics(data: SyllabusData):
    topics = parse_syllabus_into_topics(data.text)
    return {"topics": topics}


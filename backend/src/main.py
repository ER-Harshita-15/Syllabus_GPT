from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.upload import router as upload_router
from src.routes.parse_topics import router as parse_router
from src.routes.hyde_generate import router as hyde_router
from src.routes.retrieve import router as retrieve_router
from src.routes.generate_notes import router as notes_router
from src.routes.export_notes import router as export_notes_router

app = FastAPI(title="Syllabus GPT - HyDE + RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend running successfully!"}

app.include_router(upload_router, prefix="/api")
app.include_router(parse_router, prefix="/api")
app.include_router(hyde_router, prefix="/api")
app.include_router(retrieve_router, prefix="/api")
app.include_router(notes_router, prefix="/api")
app.include_router(export_notes_router, prefix="/api")


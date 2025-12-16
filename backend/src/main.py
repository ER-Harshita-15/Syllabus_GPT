from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers (ONLY ONCE EACH)
from src.routes.upload import router as upload_router
from src.routes.parse_topics import router as parse_router
from src.routes.hyde_generate import router as hyde_router
from src.routes.retrieve import router as retrieve_router
from src.routes.generate_notes import router as generate_notes_router
from src.routes.export_notes import router as export_notes_router

app = FastAPI(title="Syllabus GPT - HyDE + RAG Backend")

# ✅ Correct CORS for frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=False,                  # IMPORTANT
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend running successfully!"}

# ✅ Register routers (NO DUPLICATES)
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(parse_router, prefix="/api", tags=["Parsing"])
app.include_router(hyde_router, prefix="/api", tags=["HyDE"])
app.include_router(retrieve_router, prefix="/api", tags=["Retrieval"])
app.include_router(generate_notes_router, prefix="/api", tags=["Notes"])
app.include_router(export_notes_router, prefix="/api", tags=["Export"])

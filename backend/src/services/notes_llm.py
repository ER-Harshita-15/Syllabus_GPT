import os
from dotenv import load_dotenv
from groq import Groq

from src.services.hyde_llm import generate_hyde_document
from src.services.vector_store import retrieve_relevant_context

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env file")

client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------
# FINAL NOTES GENERATION PIPELINE
# ---------------------------------------------------------
def generate_final_notes(
    syllabus_text: str,
    subject: str = None,
    use_pyq: bool = False,
    top_k: int = 10
):
    """
    Full notes generation pipeline:
    1. Generate HYDE hypothetical doc
    2. Retrieve context from VECTOR DB
    3. Generate final structured notes
    """

    # 1. HYDE expansion
    hyde_doc = generate_hyde_document(syllabus_text)

    # 2. Retrieve chunks from Chroma
    retrieved_context = retrieve_relevant_context(
        syllabus_text=hyde_doc,
        subject=subject,
        use_pyq=use_pyq,
        top_k=top_k,
    )

    # 3. Generate final notes using Groq LLM
    system_prompt = (
        "You are a notes-writing assistant. Create beautifully structured "
        "academic notes using the given syllabus and retrieved knowledge."
    )

    user_prompt = f"""
Syllabus:
{syllabus_text}

Retrieved Context:
{retrieved_context}

Write detailed, structured notes with headings, bullet points, diagrams (ASCII), and examples.
Use clear formatting in Markdown.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # stable model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
    )

    msg = response.choices[0].message.content
    return msg

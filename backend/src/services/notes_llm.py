import os
import re
from typing import List, Dict

from dotenv import load_dotenv
from groq import Groq

from src.services.hyde_llm import generate_hyde_document
from src.services.vector_store import retrieve_relevant_context

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")

client = Groq(api_key=GROQ_API_KEY)

# Use the model that is actually working in your project
MODEL_NAME = "llama-3.3-70b-versatile"


# -------------------------------------------------
# 1. Split syllabus into UNIT-wise chunks
# -------------------------------------------------
def split_syllabus_into_units(syllabus_text: str) -> List[Dict[str, str]]:
    """
    Split a long syllabus into units:
      UNIT-I, UNIT II, UNIT-III, UNIT-IV, UNIT-V ...
    Returns: list of { "unit_title": "...", "unit_text": "..." }
    """
    # Normalize spacing a bit
    text = syllabus_text.replace("\r", " ").strip()

    # Pattern to capture markers like UNIT-I:, UNIT II:, UNIT-III:
    pattern = r"(UNIT[\-\s]*[IVXLC0-9]+:?)"

    parts = re.split(pattern, text, flags=re.IGNORECASE)

    # If no UNIT found, treat everything as one big "UNIT"
    if len(parts) == 1:
        return [{"unit_title": "UNIT", "unit_text": text}]

    units: List[Dict[str, str]] = []
    # parts looks like: ["", "UNIT-I", " text1 ...", "UNIT-II", " text2 ...", ...]
    it = iter(parts)
    first = next(it)  # prefix before first UNIT (often empty / heading) – ignore

    for marker, body in zip(it, it):
        unit_title = marker.strip()
        unit_text = body.strip()
        if not unit_text:
            continue
        units.append({"unit_title": unit_title, "unit_text": unit_text})

    return units


# -------------------------------------------------
# 2. Generate notes for a single UNIT
# -------------------------------------------------
def generate_unit_notes(
    unit_title: str,
    unit_text: str,
    subject: str | None,
    use_pyq: bool,
    top_k: int,
) -> str:
    """
    For ONE unit:
      - Use HYDE to expand the unit syllabus
      - Use RAG to fetch relevant chunks
      - Ask LLM to write detailed notes, covering ALL subtopics
    """

    # HYDE expansion for this unit
    hyde_seed = f"{subject or ''} {unit_title}: {unit_text}"
    hyde_doc = generate_hyde_document(hyde_seed)

    # Retrieve RAG context for this unit (using HYDE text as query)
    rag_context = retrieve_relevant_context(
        syllabus_text=hyde_doc,
        subject=subject,
        use_pyq=use_pyq,
        top_k=top_k,
    )

    # LLM prompt to generate *complete* unit notes
    system_prompt = (
        "You are a university-level notes generator. "
        "You must create detailed, exam-focused, textbook-style notes in Markdown."
    )

    user_prompt = f"""
You are writing notes for the subject: {subject or "Unknown Subject"}.

This is the official UNIT syllabus text. You MUST cover **every subtopic** mentioned here:

=== UNIT SYLLABUS ({unit_title}) ===
{unit_text}
==============================

You also have the following retrieved reference context (from books, notes, PYQs).
Use it to stay accurate, but you are allowed to use your own knowledge if something is missing:

=== RETRIEVED CONTEXT ===
{rag_context}
=========================

Write high-quality notes for **this unit only**.

STRICT RULES:
- Output must be in **Markdown**.
- Start with: `## {unit_title} – Detailed Notes`
- Then add logical subsections using `###` and `####`.
- Ensure you explicitly explain EVERY subtopic mentioned in the UNIT SYLLABUS.
- Use bullet points for lists.
- Add short examples where helpful.
- You MAY use general AI knowledge in addition to the context.
- Do NOT mention the words HYDE, RAG, retrieval, or context.
- Do NOT say 'according to the syllabus' – just write the notes.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# -------------------------------------------------
# 3. Generate final notes for the whole syllabus
# -------------------------------------------------
def generate_final_notes(
    syllabus_text: str,
    subject: str | None = None,
    use_pyq: bool = False,
    top_k: int = 25,
) -> str:
    """
    Main entry used by routes:
      - Splits syllabus into units
      - Generates unit-wise notes
      - Concats into a single Markdown document
    """

    # 1) Split into units
    units = split_syllabus_into_units(syllabus_text)

    if not units:
        # fallback: treat everything as one block
        units = [{"unit_title": "UNIT", "unit_text": syllabus_text}]

    # 2) For each unit, generate notes
    all_notes_blocks: List[str] = []

    for unit in units:
        unit_title = unit["unit_title"]
        unit_text = unit["unit_text"]

        try:
            unit_md = generate_unit_notes(
                unit_title=unit_title,
                unit_text=unit_text,
                subject=subject,
                use_pyq=use_pyq,
                top_k=top_k,
            )
        except Exception as e:
            # If one unit fails, add a placeholder and continue
            unit_md = f"## {unit_title} – Notes\n\n*Error generating notes for this unit: {e}*"

        all_notes_blocks.append(unit_md)

    # 3) Build final Markdown document
    subject_title = subject or "Subject"

    final_md = f"# {subject_title} – Complete Notes\n\n"

    # Add a quick index of units
    final_md += "## Units Covered\n"
    for unit in units:
        final_md += f"- {unit['unit_title']}\n"
    final_md += "\n---\n\n"

    # Append unit-wise notes
    final_md += "\n\n---\n\n".join(all_notes_blocks)

    return final_md

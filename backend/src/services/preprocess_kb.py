import os
import uuid
import re
from typing import List, Dict

import numpy as np
import easyocr
import fitz  # PyMuPDF

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ==== PATHS ====
RAW_DIR = "./knowledgebase/raw_files"
PROCESSED_DIR = "./knowledgebase/processed"
VECTOR_DB_DIR = "./vector-db"

# ==== MODELS ====
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Free embeddings
easy_reader = easyocr.Reader(['en'], gpu=False)     # OCR for scanned PDFs

# ==== CHROMA CLIENT ====
client = PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection("study_kb")


# ---------- SUBJECT DETECTION ----------
def detect_subject(filename: str) -> str:
    name = filename.lower()

    if "ai" in name:
        return "AI"
    if "ml" in name:
        return "ML"
    if "iot" in name or "internet of things" in name:
        return "IOT"
    if "toc" in name or "theory of computation" in name:
        return "TOC"
    if "stds" in name or "stats" in name or "statistics" in name:
        return "STDS"

    return "UNKNOWN"


# ---------- BOOK vs PYQ DETECTION ----------
def is_pyq(filename: str, extracted_text: str) -> bool:
    name = filename.lower()
    has_year = any(year in name for year in ["2021", "2022", "2023", "2024", "2025"])
    very_little_text = len(extracted_text.strip()) < 500  # scanned PDFs = low text

    return has_year or very_little_text


# ---------- RAW PDF TEXT EXTRACTION ----------
def extract_text_from_pdf(path: str) -> str:
    try:
        return extract_text(path)
    except Exception as e:
        print(f"[ERROR] PDFMiner failed for {path}: {e}")
        return ""


# ---------- OCR USING PYMUPDF + EASYOCR (NO POPPLER NEEDED) ----------
def extract_text_ocr(pdf_path: str) -> str:
    print(f"[OCR] Opening PDF with PyMuPDF: {pdf_path}")
    output = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Cannot open PDF via PyMuPDF: {e}")
        return ""

    for page_number in range(len(doc)):
        print(f"    - OCR on page {page_number + 1}")

        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=200)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        result = easy_reader.readtext(img, detail=0)
        output.extend(result)

    return "\n".join(output)


# ---------- CLEAN BOOK TEXT ----------
def clean_book_text(text: str) -> str:
    noise_keywords = [
        "copyright", "all rights reserved", "isbn", "publisher",
        "acknowledgements", "acknowledgments", "preface",
        "about the author", "table of contents", "contents",
        "printed in", "edition", "foreword"
    ]

    lines = text.splitlines()
    cleaned_lines = []

    for ln in lines:
        low = ln.lower().strip()
        if any(kw in low for kw in noise_keywords):
            continue
        if len(low) <= 3:
            continue
        cleaned_lines.append(ln)

    cleaned_text = "\n".join(cleaned_lines)

    FRONT_SKIP_CHARS = 3000
    if len(cleaned_text) > FRONT_SKIP_CHARS:
        cleaned_text = cleaned_text[FRONT_SKIP_CHARS:]

    return cleaned_text


# ---------- SPLIT PYQs INTO QUESTIONS ----------
def split_questions(text: str) -> List[str]:
    pattern = r"(Q\.?\s*\d+[^:.\n]*[:.]|Question\s*\d+[:.]|Q\s*\d+)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    questions = [p.strip() for p in parts if len(p.strip()) > 20]

    return questions if questions else [text]


# ---------- CHUNKING ----------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks


# ---------- BATCH INSERT ----------
def add_in_batches(documents: List[str], embeddings: List[List[float]], metadata_base: Dict, batch_size: int = 1000):
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_embeds = embeddings[start:end]
        batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
        batch_meta = [metadata_base] * len(batch_docs)

        print(f"  → Adding batch {start} to {end} ({len(batch_docs)} docs)...")

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_meta
        )


# ---------- MAIN PIPELINE ----------
def process_all_files():
    print("\n[START] Processing ALL knowledgebase files (Books + PYQs)...\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for file in os.listdir(RAW_DIR):

        if not file.lower().endswith(".pdf"):
            print(f"[SKIP] Not a PDF: {file}")
            continue

        file_path = os.path.join(RAW_DIR, file)
        print(f"\n[FILE] {file}")

        subject = detect_subject(file)
        print(f"    Subject detected → {subject}")

        raw_text = extract_text_from_pdf(file_path)
        pyq_flag = is_pyq(file, raw_text)

        if pyq_flag:
            print("    → Treating as PYQ (OCR via PyMuPDF + EasyOCR)")
            full_text = extract_text_ocr(file_path)
            content_type = "PYQ"
        else:
            print("    → Treating as BOOK (PDFMiner text)")
            if len(raw_text.strip()) < 50:
                print("    [WARNING] Too little text for book — skipping.")
                continue
            full_text = clean_book_text(raw_text)
            content_type = "BOOK"

        if not full_text or len(full_text.strip()) < 50:
            print("    [WARNING] No usable text — skipping.")
            continue

        processed_path = os.path.join(
            PROCESSED_DIR,
            f"{os.path.splitext(file)[0]}_{content_type}.txt"
        )

        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        if content_type == "PYQ":
            print("    Splitting into questions...")
            questions = split_questions(full_text)
            print(f"    → Found ~{len(questions)} questions")
            documents = []
            for q in questions:
                documents.extend(chunk_text(q))
        else:
            documents = chunk_text(full_text)

        print(f"    Total chunks → {len(documents)}")

        if len(documents) == 0:
            print("    No chunks — skipping.")
            continue

        try:
            collection.delete(where={"source": file, "type": content_type})
        except:
            pass

        print("    Embedding chunks...")
        embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

        metadata_base = {
            "source": file,
            "type": content_type,
            "subject": subject,
        }

        print("    Storing in ChromaDB...")
        add_in_batches(documents, embeddings, metadata_base)

    print("\n[DONE] KB processing complete! 🚀\n")


if __name__ == "__main__":
    process_all_files()

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# === Paths ===
VECTOR_DB_DIR = "./vector-db"

# === Embedding Model ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # must match preprocess embeddings

# === ChromaDB Client ===
client = PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection("study_kb")


# ---------------------------------------------------------
#  BASIC RAW VECTOR SEARCH  (needed for /query route)
# ---------------------------------------------------------
def vector_search(query: str, top_k: int = 5):
    """Raw semantic search without filters."""
    
    query_embedding = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# ---------------------------------------------------------
#  FILTERED CONTEXT RETRIEVAL  (used for notes generation)
# ---------------------------------------------------------
def retrieve_relevant_context(
        syllabus_text: str,
        subject: str = None,
        use_pyq: bool = False,
        top_k: int = 10
    ):
    """
    Retrieves the most relevant BOOK or PYQ chunks based on the given syllabus text.
    """

    ### Fix: Chroma expects only ONE operator in "where"
    ### So we use a nested operator "$and"
    
    where_filter = {"$and": []}

    if subject and subject != "ALL":
        where_filter["$and"].append({"subject": subject})

    content_type = "PYQ" if use_pyq else "BOOK"
    where_filter["$and"].append({"type": content_type})

    # If only one filter → unwrap
    if len(where_filter["$and"]) == 1:
        where_filter = where_filter["$and"][0]

    query_embedding = embedder.encode([syllabus_text])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    docs = results.get("documents", [[]])[0]

    return "\n\n".join(docs)

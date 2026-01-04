from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_DIR = BASE_DIR / "data" / "vectorstore" / "protocols_faiss"

def get_retriever(k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": k})

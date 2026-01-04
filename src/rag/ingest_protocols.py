from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parents[2]
PROTOCOLS_DIR = BASE_DIR / "data" / "synthetic" / "protocols"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore" / "protocols_faiss"

def main():
    # 1) Carregar todos os .md
    docs = []
    for md_file in PROTOCOLS_DIR.glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        loaded = loader.load()
        # Guardar o nome do arquivo como "fonte"
        for d in loaded:
            d.metadata["source"] = md_file.name
        docs.extend(loaded)

    if not docs:
        raise RuntimeError(f"Nenhum protocolo encontrado em: {PROTOCOLS_DIR}")

    # 2) Quebrar em pedaços (chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    # 3) Embeddings (modelo leve; funciona bem local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4) Criar base vetorial FAISS
    db = FAISS.from_documents(chunks, embeddings)

    VECTOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTOR_DIR))

    print(f"✅ Protocolos indexados: {len(docs)} documentos, {len(chunks)} chunks")
    print(f"✅ Base vetorial salva em: {VECTOR_DIR}")

if __name__ == "__main__":
    main()

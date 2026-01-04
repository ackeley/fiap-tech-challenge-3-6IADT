from src.rag.retriever import get_retriever

if __name__ == "__main__":
    retriever = get_retriever(k=3)

    # Em versões novas do LangChain, use invoke()
    docs = retriever.invoke("Quais sinais de gravidade em pneumonia?")

    for i, d in enumerate(docs, 1):
        print(f"\n--- Trecho {i} | Fonte: {d.metadata.get('source')} ---")
        print(d.page_content[:600])

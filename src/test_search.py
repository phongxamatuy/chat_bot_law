from pathlib import Path
from embedding.providers import HuggingFaceEmbeddingProvider
from vectordb.vector_store import ChromaDBStore
from retrieval.retriever import AdvancedRetriever
from retrieval.parent_child_store import ParentChildStore
import pickle

def main():
    print("[1/3] Đang khởi tạo Embedding Model (CPU - Model nhẹ)...")
    provider = HuggingFaceEmbeddingProvider(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    embedder = provider.get_embedding()

    print("[2/3] Đang kết nối tới ChromaDB...")
    project_root = Path(__file__).parent.parent
    db_path = str(project_root / "storage" / "chroma_db")
    
    vector_store = ChromaDBStore(
        embedding_model=embedder,
        persist_directory=db_path
    )

    print("[3/3] Đang tải BM25, DocStore và cấu hình Advanced Retriever...")
    bm25_path = project_root / "storage" / "bm25_index.pkl"
    bm25_retriever = None
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)

    docstore_path = project_root / "storage" / "docstore.json"
    parent_store = ParentChildStore(str(docstore_path))

    retriever = AdvancedRetriever(
        vector_db_client=vector_store,
        bm25_retriever=bm25_retriever,
        embedding_model=embedder,
        top_k=5,
        fetch_k=20,
        parent_store=parent_store
    )

    # ---------------------------------------------------------
    # GIAI ĐOẠN TÌM KIẾM
    # ---------------------------------------------------------
    print("\n===============================")
    print("HỆ THỐNG ĐÃ SẴN SÀNG TÌM KIẾM")
    print("===============================\n")

    query = "Điều kiện tha tù trước thời hạn là gì?"
    print(f"Câu hỏi: '{query}'")
    
    results = retriever.retrieve(query=query)

    for i, res in enumerate(results):
        print(f"\n--- Kết quả {i+1} (Score: {res['score']:.4f}) ---")
        meta = res['metadata']
        law = meta.get('law', '')
        article = meta.get('article', '')
        level = meta.get('level', '')
        
        print(f"Location: {law} -> {article} (Level: {level})")
        print(f"[CHILD CONTENT]:\n{res['content'][:400]}...")
        
        parent = res.get('parent_content')
        if parent:
            print(f"\n[PARENT CONTENT]:\n{parent[:400]}...")

if __name__ == "__main__":
    main()

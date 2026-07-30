from ingestion.loader import CleanTextLoader
from pathlib import Path
from parsing.law_parser import LawParser
from chunking.hierarchical_chunker import HierarchicalChunker
from retrieval.parent_child_store import ParentChildStore
from embedding.providers import HuggingFaceEmbeddingProvider
from vectordb.vector_store import ChromaDBStore
import pickle
from langchain_community.retrievers import BM25Retriever
import os

# Lấy đường dẫn gốc của project
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "processed" / "final"

print("[1/5] Đang load dữ liệu text từ thư mục final...")
loader = CleanTextLoader(str(data_path))
documents = loader.load()

print("[2/5] Đang phân tích cú pháp (Parsing) và chia chunk phân cấp (Hierarchical Chunking)...")
parser = LawParser()
chunker = HierarchicalChunker(chunk_size=1500, chunk_overlap=200)

all_chunks = []
all_parent_docs = {}

for doc in documents:
    source = doc.metadata.get('source', '')
    file_name = os.path.basename(source)
    
    # Bóc tách
    nodes = parser.parse(doc.page_content, file_name)
    articles = parser.extract_all_articles(nodes)
    
    # Chia chunk
    chunks, parent_docs = chunker.chunk(articles, file_name)
    all_chunks.extend(chunks)
    all_parent_docs.update(parent_docs)

print(f"[*] Tổng số chunk: {len(all_chunks)}")
print(f"[*] Tổng số parent docs: {len(all_parent_docs)}")

print("[3/5] Đang lưu Parent-Child Docs vào DocStore...")
store_path = project_root / "storage" / "docstore.json"
parent_store = ParentChildStore(str(store_path))
parent_store.save(all_parent_docs)

print("[4/5] Đang khởi tạo Embedding Model và lưu Vector vào ChromaDB...")
provider = HuggingFaceEmbeddingProvider(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)
embedder = provider.get_embedding()

db_path = str(project_root / "storage" / "chroma_db")
vector_store = ChromaDBStore(
    embedding_model=embedder,
    persist_directory=db_path
)

# Chú ý: ChromaDBStore có hàm add_documents
vector_store.add_documents(all_chunks, batch_size=100)

print("[5/5] Đang tạo và lưu Keyword Index (BM25)...")
bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_path = project_root / "storage" / "bm25_index.pkl"
with open(bm25_path, "wb") as f:
    pickle.dump(bm25_retriever, f)
print(f"[*] Đã lưu BM25 index tại: {bm25_path}")

print("\n===============================")
print("Quá trình nạp Vector Database, BM25 và Parent-Child DocStore hoàn tất!")

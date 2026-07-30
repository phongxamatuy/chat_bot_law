from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

class ChromaDBStore:
    def __init__(
        self, 
        embedding_model: Embeddings,
        persist_directory: str = "storage/chroma_db",
        collection_name: str = "law_chatbot"
    ):
        """
        Khởi tạo ChromaDB Store.
        
        :param embedding_model: Object Embeddings dùng để nhúng text.
        :param persist_directory: Đường dẫn lưu trữ DB cố định.
        :param collection_name: Tên collection.
        """
        self.embedding_model = embedding_model
        
        # Đảm bảo thư mục lưu trữ tồn tại
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory
        )

    def _generate_id(self, doc: Document) -> str:
        """
        Lấy chunk_id từ metadata. Nếu không có (như trường hợp tự thêm text ngoài), 
        thì tạo ID định danh duy nhất dựa trên tên file và nội dung.
        """
        if "chunk_id" in doc.metadata:
            return doc.metadata["chunk_id"]
            
        source = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
        content = doc.page_content
        unique_string = f"{source}:::{content}"
        return hashlib.md5(unique_string.encode("utf-8")).hexdigest()

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """
        Thêm (Upsert) các Document vào ChromaDB theo từng batch.
        Nếu ID đã tồn tại, nó sẽ Ghi Đè (Overwrite) thay vì tạo mới (nhân đôi).
        """
        print(f"[*] Đang thêm/cập nhật {len(documents)} documents vào ChromaDB...")
        
        # Lưu vết các ID đã sinh ra trên toàn bộ quá trình để tránh trùng lặp
        global_seen_ids = set()
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = []
            
            for doc in batch:
                base_id = self._generate_id(doc)
                final_id = base_id
                counter = 1
                while final_id in global_seen_ids:
                    final_id = f"{base_id}_{counter}"
                    counter += 1
                global_seen_ids.add(final_id)
                batch_ids.append(final_id)
            
            # Pass ids vào hàm add_documents để kích hoạt tính năng Upsert
            self.vector_store.add_documents(documents=batch, ids=batch_ids)
            print(f"  - Đã xử lý batch {i // batch_size + 1} ({len(batch)} chunks)")
            
        print("[*] Đã lưu xong vào ChromaDB.")

    def similarity_search(
        self, 
        query: str, 
        k: int = 3, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Tìm kiếm các Document gần nhất với câu query.
        """
        return self.vector_store.similarity_search(query=query, k=k, filter=filter)

    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        """
        Trả về Retriever interface của Langchain để kết hợp với các chuỗi (chains).
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

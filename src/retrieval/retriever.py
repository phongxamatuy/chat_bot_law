from typing import List, Dict, Any
import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from sentence_transformers import CrossEncoder

class AdvancedRetriever:
    def __init__(
        self, 
        vector_db_client, 
        bm25_retriever, 
        embedding_model, 
        top_k: int = 5, 
        fetch_k: int = 20,
        parent_store = None
    ):
        """
        Khởi tạo Advanced Retriever tích hợp Hybrid (RRF), MMR, Re-ranking và Parent-Child.
        """
        self.vector_db_client = vector_db_client
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.parent_store = parent_store
        
        # 1. Cấu hình Vector Retriever (Chroma)
        self.vector_retriever = self.vector_db_client.as_retriever(search_kwargs={"k": self.fetch_k})
        
        # 2. Cấu hình BM25 Retriever và Hybrid
        if self.bm25_retriever:
            self.bm25_retriever.k = self.fetch_k
            
            # Tạo Ensemble (Hybrid) sử dụng thuật toán Reciprocal Rank Fusion (RRF)
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]
            )
        else:
            self.hybrid_retriever = self.vector_retriever

        # 3. Tải mô hình Re-ranker (Cross-Encoder) siêu nhẹ
        print("\n[*] Đang tải mô hình Re-ranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')

    def retrieve(self, query: str, top_k: int = None, filters: Dict[str, Any] = None, threshold: float = -5.0) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k
        
        print(f"\n[*] BƯỚC 1: Hybrid Search (BM25 + Semantic) - Lấy top {self.fetch_k} chunks...")
        docs = self.hybrid_retriever.invoke(query)
        
        if not docs:
            return []
            
        print(f"[*] BƯỚC 2: Tính toán đa dạng (MMR)...")
        # Encode lại vector cho các docs vừa tìm được (cực nhanh vì số lượng bé)
        doc_texts = [d.page_content for d in docs]
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
        query_embedding = self.embedding_model.embed_query(query)
        
        # Chạy thuật toán MMR để loại bỏ các đoạn text bị lặp ý (giảm từ 20 xuống 15 chunks)
        mmr_k = min(15, len(docs))
        mmr_indices = maximal_marginal_relevance(
            np.array(query_embedding),
            doc_embeddings,
            lambda_mult=0.5, # lambda=0.5: Tỉ lệ 50/50 giữa Relevance và Diversity
            k=mmr_k
        )
        mmr_docs = [docs[i] for i in mmr_indices]
        
        print(f"[*] BƯỚC 3: Re-ranking bằng Cross-Encoder (chấm điểm lại {len(mmr_docs)} chunks)...")
        # Gói câu hỏi và nội dung thành từng cặp để chấm điểm chéo
        pairs = [[query, d.page_content] for d in mmr_docs]
        scores = self.reranker.predict(pairs)
        
        # Ghép score vào docs và sort giảm dần
        scored_docs = list(zip(mmr_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"[*] BƯỚC 4: Lọc Threshold và xuất Top {k}...")
        final_results = []
        for doc, score in scored_docs:
            if score >= threshold:
                final_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
                if len(final_results) == k:
                    break
                    
        if self.parent_store and final_results:
            final_results = self.parent_store.get_parents_for_chunks(final_results)
            print(f"[*] BƯỚC 5: Parent Expansion - Đã bổ sung ngữ cảnh cha cho {len(final_results)} chunks...")
            
        return final_results

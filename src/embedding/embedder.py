from typing import List

from langchain_core.documents import Document

from .providers import EmbeddingProvider


class Embedder:

    def __init__(
        self,
        provider: EmbeddingProvider
    ):
        self.embedding = provider.get_embedding()

    def embed_documents(
        self,
        documents: List[Document]
    ):

        texts = [
            doc.page_content
            for doc in documents
        ]

        return self.embedding.embed_documents(texts)

    def embed_query(
        self,
        query: str
    ):

        return self.embedding.embed_query(query)
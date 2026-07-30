from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingProvider(ABC):
    """
    Interface cho mọi embedding model.
    """

    @abstractmethod
    def get_embedding(self) -> Embeddings:
        pass


class HuggingFaceEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu"
    ):

        self.model_name = model_name
        self.device = device

    def get_embedding(self):

        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.device
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )
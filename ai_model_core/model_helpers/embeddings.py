# Standard library imports
from typing import List

# Third-party imports
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class E5Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        normalize_embeddings: bool = True
    ):
        """
        Initialize E5 embeddings for retrieval tasks.
        
        Args:
            model_name: The name of the E5 model to use
            normalize_embeddings: Whether to normalize the embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

    def _format_text(self, text: str, is_query: bool = False) -> str:
        """
        Format text according to E5 rules for retrieval:
        - "query: " for questions
        - "passage: " for documents
        """
        prefix = "query: " if is_query else "passage: "
        return prefix + text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using 'passage:' prefix."""
        formatted_texts = [
            self._format_text(text, is_query=False) for text in texts
        ]
        embeddings = self.model.encode(
            formatted_texts,
            normalize_embeddings=self.normalize_embeddings
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query using 'query:' prefix."""
        formatted_text = self._format_text(text, is_query=True)
        embedding = self.model.encode(
            formatted_text,
            normalize_embeddings=self.normalize_embeddings
        )
        return embedding.tolist()

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRetriever:
    """Pure dense embedding retriever."""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.chunk_embeddings = None
        print(f"Loaded embedding model: {embedding_model}")
    
    def build(self, chunks: List[str]):
        """Build embedding index."""
        self.chunks = chunks
        print(f"Computing embeddings for {len(chunks)} chunks...")
        self.chunk_embeddings = self.embedding_model.encode(
            chunks, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Embedding index built: {self.chunk_embeddings.shape}")
    
    def search(self, query: str, topk: int = 5) -> List[Tuple[float, str]]:
        """Search using cosine similarity."""
        if not self.chunks or self.chunk_embeddings is None:
            return []
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Scale similarities to 0-1 range for better thresholding
        # Cosine similarity is already 0-1, but let's enhance the range
        scaled_scores = similarities
        
        # Create results
        results = [(score, chunk) for score, chunk in zip(scaled_scores, self.chunks)]
        
        # Sort and return top-k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:topk]
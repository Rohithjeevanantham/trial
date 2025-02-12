import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGRetriever:
    def __init__(self, index_path='embeddings.faiss', 
                 mapping_path='chunk_mapping.json',
                 model_name='all-MiniLM-L6-v2'):
        """Initialize the retriever with pre-built index and mapping."""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
            
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
    
    def _create_query_embedding(self, query):
        """Create embedding for the query."""
        return self.model.encode([query])[0].reshape(1, -1).astype('float32')
    
    def retrieve(self, query, k=3):
        """
        Retrieve the k most similar chunks for the query.
        
        Args:
            query (str): The query text
            k (int): Number of chunks to retrieve
            
        Returns:
            list: List of dictionaries containing retrieved chunks and their scores
        """
        # Create query embedding
        query_embedding = self._create_query_embedding(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            chunk = self.mapping[str(idx)]
            results.append({
                'chunk': chunk,
                'score': float(score),  # Convert numpy float to Python float
                'main_heading': chunk['main_heading'],
                'text': chunk['text']
            })
        
        return results
    
    def get_relevant_context(self, query, k=3):
        """
        Get relevant context as a formatted string.
        
        Args:
            query (str): The query text
            k (int): Number of chunks to retrieve
            
        Returns:
            str: Formatted context string
        """
        results = self.retrieve(query, k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_part = (
                f"[Document {i}]\n"
                f"Topic: {result['main_heading']}\n"
                f"{result['text']}\n"
                f"Relevance Score: {result['score']:.4f}\n"
            )
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

def main():
    # Example usage
    retriever = RAGRetriever()
    
    # Test queries
    test_queries = [
        "What's the status of 3D Bioprinting?",
        "Tell me about completed projects",
    ]
    
    print("🔍 Testing Retrieval System")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("\nRetrieved Context:")
        print(retriever.get_relevant_context(query))
        print("-" * 50)

if __name__ == "__main__":
    main()

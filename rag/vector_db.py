import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import uuid

from rag import embed_texts


class VectorDB:
    def __init__(self, dim: int, collection_name: str = "grammar_rules"):
        """
        Initialize VectorDB with Qdrant.
        
        Args:
            dim (int): Dimension of the vectors
            collection_name (str): Name of the Qdrant collection
        """
        self.dim = dim
        self.collection_name = collection_name
        
        # Initialize Qdrant client (in-memory for simplicity)
        self.client = QdrantClient(":memory:")
        
        # Create collection if it doesn't exist
        self._create_collection()
        
        # Keep track of points for easy access
        self.points = {}  # Map point_id to text

    def _create_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    def add_vector(self, vector: np.ndarray, text: str):
        """
        Add a vector to the store.

        Args:
            vector (numpy.ndarray): The vector embedding to be stored.
            text (str): The corresponding text.
        """
        vector = self._normalize(vector).astype('float32')
        point_id = str(uuid.uuid4())
        
        # Add to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={"text": text}
                )
            ]
        )
        
        # Store mapping
        self.points[point_id] = text

    def add_chunks(self, chunks: list[str], emb_model: str):
        """
        Adds multiple chunks to the DB, embedding them first.

        Args:
            chunks (list[str]): The list of chunks to be stored.
            emb_model: The embedding model to be used for embedding
        """
        embeddings = embed_texts(chunks, emb_model)

        for vector, text in zip(embeddings, chunks):
            self.add_vector(vector, text)

    def get_vector(self, point_id: str):
        """
        Retrieve a vector from the store.

        Args:
            point_id (str): The identifier of the vector to retrieve.
        Returns:
            np.ndarray: The vector data if found, or None if not found.
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            if result:
                return np.array(result[0].vector)
            return None
        except:
            return None

    def search(self, query_vector: np.ndarray, num_results: int = 5):
        """
        Find similar vectors to the query vector.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            List[dict]: Each dict contains the point_id, text, and similarity score.
        """
        query_vector = self._normalize(query_vector).astype('float32')
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=num_results,
            with_payload=True
        )
        
        results = []
        for point in search_result:
            results.append({
                "id": point.id,
                "text": point.payload["text"],
                "score": float(point.score)
            })
        
        return results

    def clear(self):
        """Clear all vectors from the database."""
        try:
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            self.points.clear()
        except:
            pass

    def get_collection_info(self):
        """Get information about the collection."""
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return None
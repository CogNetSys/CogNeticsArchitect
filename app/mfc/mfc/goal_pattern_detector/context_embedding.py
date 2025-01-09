# mfc/mfc/goal_pattern_detector/context_embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np
from collections import OrderedDict
import logging

class ContextEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2', max_cache_size=10000, max_length=512):
        """
        Initializes the ContextEmbedding class with the specified model.

        Args:
            model_name (str): Name of the pre-trained Sentence-BERT model.
            max_cache_size (int): Maximum number of context embeddings to cache.
            max_length (int): Maximum token length for the model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.max_length = max_length  # Maximum token length for the model

        logging.info(f"Initialized ContextEmbedding with model {self.model_name}")

    def get_embedding(self, context: str) -> np.ndarray:
        """
        Retrieves the embedding for the given context. If the embedding is cached, it returns the cached value.
        Otherwise, it generates a new embedding, caches it, and returns it.

        Args:
            context (str): The context string to embed.

        Returns:
            np.ndarray: The embedding vector.
        """
        if context in self.cache:
            # Move to end to indicate recent use
            self.cache.move_to_end(context)
            logging.debug("Retrieved embedding from cache.")
            return self.cache[context]

        # Handle context length by truncation if necessary
        if len(context.split()) > self.max_length:
            context = ' '.join(context.split()[:self.max_length])
            logging.warning("Context truncated to max_length.")

        # Generate embedding
        try:
            embedding = self.model.encode(context)
            logging.debug("Generated new embedding.")
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return np.array([])  # Return empty array on failure

        # Cache the embedding
        if len(self.cache) >= self.max_cache_size:
            # Remove the first (least recently used) item
            removed = self.cache.popitem(last=False)
            logging.debug(f"Cache full. Removed least recently used context: {removed[0]}")

        self.cache[context] = embedding
        logging.debug("Cached new embedding.")

        return embedding

    def calculate_similarity(self, context1: np.ndarray, context2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two context embeddings.

        Args:
            context1 (np.ndarray): Embedding of the first context.
            context2 (np.ndarray): Embedding of the second context.

        Returns:
            float: Cosine similarity score.
        """
        if context1.size == 0 or context2.size == 0:
            logging.warning("One or both embeddings are empty.")
            return 0.0
        similarity = np.dot(context1, context2) / (np.linalg.norm(context1) * np.linalg.norm(context2))
        logging.debug(f"Calculated similarity: {similarity}")
        return similarity

    def summarize_context(self, context: str) -> str:
        """
        Summarizes the context if it exceeds the maximum length using a simple heuristic.

        Args:
            context (str): The context string to summarize.

        Returns:
            str: The summarized context.
        """
        # Placeholder for summarization logic. This can be enhanced using an LLM or a summarization model.
        words = context.split()
        if len(words) > self.max_length:
            summarized = ' '.join(words[:self.max_length]) + "..."
            logging.info("Context summarized by truncation.")
            return summarized
        return context

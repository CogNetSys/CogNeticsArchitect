# mfc/mfc/goal_pattern_detector/tests/test_context_embedding.py
# test with pythons -m unittest discover -s mfc/mfc/goal_pattern_detector/tests
import unittest
from goal_pattern_detector.context_embedding import ContextEmbedding

class TestContextEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = ContextEmbedding(max_cache_size=2)

    def test_get_embedding_caching(self):
        context1 = "This is a test context."
        context2 = "Another context for embedding."
        embedding1 = self.embedding.get_embedding(context1)
        embedding2 = self.embedding.get_embedding(context2)
        self.assertIn(context1, self.embedding.cache)
        self.assertIn(context2, self.embedding.cache)

        # Access context1 again to make it recently used
        embedding1_again = self.embedding.get_embedding(context1)
        self.assertEqual(embedding1.all(), embedding1_again.all())

        # Add a third context to exceed cache size
        context3 = "Third context to exceed cache."
        embedding3 = self.embedding.get_embedding(context3)
        self.assertNotIn(context2, self.embedding.cache)  # Least recently used should be removed
        self.assertIn(context1, self.embedding.cache)
        self.assertIn(context3, self.embedding.cache)

    def test_calculate_similarity(self):
        context1 = "Context one."
        context2 = "Context two."
        embedding1 = self.embedding.get_embedding(context1)
        embedding2 = self.embedding.get_embedding(context2)
        similarity = self.embedding.calculate_similarity(embedding1, embedding2)
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)

    def test_summarize_context(self):
        long_context = "word " * 600  # Assuming max_length=512 tokens
        summarized = self.embedding.summarize_context(long_context)
        self.assertTrue(len(self.embedding.tokenizer.encode(summarized)) <= self.embedding.max_length)

if __name__ == '__main__':
    unittest.main()
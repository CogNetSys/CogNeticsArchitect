# test_node_encoder.py

import unittest
from node_encoder import NodeEncoder
import numpy as np

class TestNodeEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = NodeEncoder()
        self.agent_data = {
            "Base Model": "GPT-4",
            "Role/Expertise": "Programmer",
            "Current State": {
                "Task": 3,
                "Resources": 5
            },
            "Available Plugins/Tools": "Python Compiler"
        }
        self.task_data = {
            "Type": "Development",
            "Resource Requirements": {
                "CPU": 4,
                "Memory": 16,
                "Storage": 100
            },
            "Deadline": "2025-02-01",
            "Dependencies": "Task 2",
            "Priority": "High"
        }

    def test_encode_agent(self):
        embedding = self.encoder.encode_agent(self.agent_data)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 384 + 2)  # all-MiniLM-L6-v2 outputs 384-dim embeddings

    def test_encode_task(self):
        embedding = self.encoder.encode_task(self.task_data)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 384 + 3)  # all-MiniLM-L6-v2 outputs 384-dim embeddings

    def test_encode_graph(self):
        graph_data = {}
        embedding = self.encoder.encode_graph(graph_data)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 0)  # Placeholder implementation

if __name__ == '__main__':
    unittest.main()
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
            "Available Plugins/Tools": ["Python Compiler", "File Reader"],
            "Expertise Level": "Intermediate",
            "Current Workload": 4,
            "Reliability Score": 0.95,
            "Latency": 0.2,
            "Error Rate": 0.01,
            "Cost Per Task": 0.5
        }
        self.task_data = {
            "Type": "Development",
            "Resource Requirements": {
                "CPU": 4,
                "Memory": 16,
                "Storage": 100
            },
            "Deadline": "2025-02-01",
            "Dependencies": ["Task 2"],
            "Priority": "High",
            "Computational Complexity": 3,
            "Memory Footprint": 8,
            "Data Locality": "Local",
            "Security Level": "Confidential",
            "Urgency Score": 0.9,
            "Expected Value": 0.8,
            "Precedence Relations": ["Task1 -> Task2"]
        }

    def test_encode_agent(self):
        embedding = self.encoder.encode_agent(self.agent_data)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 384 + 19)  # Adjust based on encoding scheme

    def test_encode_task(self):
        embedding = self.encoder.encode_task(self.task_data)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 384 + 26)  # Adjust based on encoding scheme

    def test_encode_dependencies(self):
        dependencies = ["Task 1", "Task 2"]
        encoded_dependencies = self.encoder.encode_dependencies(dependencies)
        self.assertIsInstance(encoded_dependencies, np.ndarray)
        self.assertEqual(len(encoded_dependencies), 5)

    def test_encode_precedence_relations(self):
        relations = ["Task1 -> Task2", "Task2 -> Task3"]
        encoded_relations = self.encoder.encode_precedence_relations(relations)
        self.assertIsInstance(encoded_relations, np.ndarray)
        self.assertEqual(len(encoded_relations), 20)  # 2 values per relation, max 10 relations

    def test_one_hot_encode(self):
        categories = ["Cat", "Dog", "Bird"]
        value = "Dog"
        encoded = self.encoder.one_hot_encode(value, categories)
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(len(encoded), 3)
        self.assertTrue(np.array_equal(encoded, np.array([0, 1, 0])))

    def test_encode_graph(self):
        graph_data = {}  # Add a sample graph
        embedding = self.encoder.encode_graph(graph_data)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 0)  # Placeholder implementation

if __name__ == '__main__':
    unittest.main()
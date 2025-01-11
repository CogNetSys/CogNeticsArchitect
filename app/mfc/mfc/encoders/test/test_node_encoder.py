import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from encoders.node_encoder import NodeEncoder
import numpy as np

@pytest.fixture
def encoder():
    return NodeEncoder()

def test_encode_agent(encoder):
    agent_data = {
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
    embedding = encoder.encode_agent(agent_data)
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 413  # Corrected expected length

def test_encode_task(encoder):
    task_data = {
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
    embedding = encoder.encode_task(task_data)
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 436  # Corrected expected length

def test_encode_dependencies(encoder):
    dependencies = ["Task 1", "Task 2"]
    encoded_dependencies = encoder.encode_dependencies(dependencies)
    assert isinstance(encoded_dependencies, np.ndarray)
    assert len(encoded_dependencies) == 5

def test_encode_precedence_relations(encoder):
    relations = ["Task1 -> Task2", "Task2 -> Task3"]
    encoded_relations = encoder.encode_precedence_relations(relations)
    assert isinstance(encoded_relations, np.ndarray)
    assert len(encoded_relations) == 20

def test_one_hot_encode(encoder):
    categories = ["Cat", "Dog", "Bird"]
    value = "Dog"
    encoded = encoder.one_hot_encode(value, categories)
    assert isinstance(encoded, np.ndarray)
    assert len(encoded) == 3
    assert np.array_equal(encoded, np.array([0, 1, 0]))

def test_encode_graph(encoder):
    graph_data = {}
    embedding = encoder.encode_graph(graph_data)
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 0
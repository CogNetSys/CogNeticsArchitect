import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from encoders.node_encoder import NodeEncoder

def test_encode_agent():
    encoder = NodeEncoder()
    agent_data = {
        "Base Model": "GPT-4",
        "Role/Expertise": "Programmer",
        "Available Plugins/Tools": ["Python Compiler", "Web Searcher"],
        "Expertise Level": "Expert",
        "Current State": {"Task": 5, "Resources": 3},
        "Current Workload": 7,
        "Reliability Score": 0.95,
        "Latency": 0.2,
        "Error Rate": 0.01,
        "Cost Per Task": 0.5,
    }
    embedding = encoder.encode_agent(agent_data)
    assert embedding is not None
    assert len(embedding) > 0  # Ensure the embedding is not empty

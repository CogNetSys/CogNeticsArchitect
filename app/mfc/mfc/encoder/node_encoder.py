# node_encoder.py

from sentence_transformers import SentenceTransformer
import numpy as np
import json

class NodeEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the NodeEncoder with a specified SentenceTransformer model.

        Args:
            model_name (str): The name of the pretrained SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)

    def encode_agent(self, agent_data):
        """
        Encodes agent features into a numerical embedding.

        Args:
            agent_data (dict): Dictionary containing agent features.

        Returns:
            np.ndarray: Numerical embedding of the agent.
        """
        text_features = f"{agent_data['Base Model']} {agent_data['Role/Expertise']} {agent_data['Available Plugins/Tools']}"
        text_embedding = self.model.encode(text_features)

        numerical_features = np.array([
            agent_data['Current State']['Task'],
            agent_data['Current State']['Resources']
        ])

        # One-hot encode categorical features if necessary
        # For simplicity, assuming numerical_features are already numerical

        # Combine embeddings
        combined_embedding = np.concatenate((text_embedding, numerical_features))
        return combined_embedding

    def encode_task(self, task_data):
        """
        Encodes task features into a numerical embedding.

        Args:
            task_data (dict): Dictionary containing task features.

        Returns:
            np.ndarray: Numerical embedding of the task.
        """
        text_features = f"{task_data['Type']} {task_data['Priority']} {task_data['Dependencies']}"
        text_embedding = self.model.encode(text_features)

        numerical_features = np.array([
            task_data['Resource Requirements']['CPU'],
            task_data['Resource Requirements']['Memory'],
            task_data['Resource Requirements']['Storage']
        ])

        # Combine embeddings
        combined_embedding = np.concatenate((text_embedding, numerical_features))
        return combined_embedding

    def encode_graph(self, graph_data):
        """
        Encodes graph-level features if needed.

        Args:
            graph_data (dict): Dictionary containing graph features.

        Returns:
            np.ndarray: Numerical embedding of the graph.
        """
        # Placeholder for graph encoding logic
        # Implement graph-level encoding if required
        return np.array([])

# Example usage
if __name__ == "__main__":
    encoder = NodeEncoder()

    agent_example = {
        "Base Model": "GPT-4",
        "Role/Expertise": "Programmer",
        "Current State": {
            "Task": 3,
            "Resources": 5
        },
        "Available Plugins/Tools": "Python Compiler"
    }

    task_example = {
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

    agent_embedding = encoder.encode_agent(agent_example)
    task_embedding = encoder.encode_task(task_example)

    print("Agent Embedding:", agent_embedding)
    print("Task Embedding:", task_embedding)
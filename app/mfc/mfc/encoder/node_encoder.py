import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from collections import OrderedDict

class NodeEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name, device='cpu')  # Use CPU to avoid CUDA issues
        self.model.eval()
        self.cache = OrderedDict()
        self.max_cache_size = 10000

    def encode_agent(self, agent_data):
        with torch.no_grad():
            text_features = f"{agent_data['Base Model']} {agent_data['Role/Expertise']} {agent_data['Available Plugins/Tools']}"
            text_embedding = self.model.encode(text_features, convert_to_numpy=True)

        base_model_onehot = self.one_hot_encode(agent_data['Base Model'], ['GPT-4', 'LLaMA-2', 'Mistral-7B', 'Mixtral-8x7B', 'LLaMA-3'])
        role_onehot = self.one_hot_encode(agent_data['Role/Expertise'], ['Programmer', 'Mathematician', 'Data Analyst', 'Domain Expert', 'QA Tester', 'Manager'])
        expertise_onehot = self.one_hot_encode(agent_data['Expertise Level'], ['Novice', 'Intermediate', 'Advanced', 'Expert'])

        tools_all = ['Python Compiler', 'Database Connector', 'Web Searcher', 'Image Analyzer', 'File Reader', 'Mathematical calculator', 'Text-to-speech translator']
        tools_multi_hot = [1 if tool in agent_data['Available Plugins/Tools'] else 0 for tool in tools_all]

        numerical_features = np.array([
            agent_data['Current State']['Task'],
            agent_data['Current State']['Resources'],
            agent_data['Current Workload'],
            agent_data['Reliability Score'],
            agent_data['Latency'],
            agent_data['Error Rate'],
            agent_data['Cost Per Task']
        ], dtype=float)

        combined_embedding = np.concatenate((
            text_embedding,
            base_model_onehot,
            role_onehot,
            expertise_onehot,
            np.array(tools_multi_hot),
            numerical_features
        ))
        print(f"Text Embedding Length: {len(text_embedding)}")
        print(f"Base Model One-Hot Encoding Length: {len(base_model_onehot)}")
        print(f"Role One-Hot Encoding Length: {len(role_onehot)}")
        print(f"Expertise One-Hot Encoding Length: {len(expertise_onehot)}")
        print(f"Tools Multi-Hot Encoding Length: {len(tools_multi_hot)}")
        print(f"Numerical Features Length: {len(numerical_features)}")
        print(f"Total Combined Embedding Length (Agent): {len(combined_embedding)}")

        return combined_embedding


    def encode_task(self, task_data):
        with torch.no_grad():
            text_features = f"{task_data['Type']} {task_data['Priority']} {task_data['Dependencies']}"
            text_embedding = self.model.encode(text_features, convert_to_numpy=True)

        type_onehot = self.one_hot_encode(task_data['Type'], ['Code Generation', 'Code Debugging', 'Code Optimization', 'Data Analysis', 'Text Summarization', 'Question Answering', 'Document Drafting', 'Image Generation', 'Logical Reasoning', 'Mathematical Computation', 'Information Retrieval'])
        priority_onehot = self.one_hot_encode(task_data['Priority'], ['High', 'Medium', 'Low'])
        locality_onehot = self.one_hot_encode(task_data['Data Locality'], ['Local', 'Edge', 'Cloud'])
        security_onehot = self.one_hot_encode(task_data['Security Level'], ['Confidential', 'Restricted', 'Public'])

        numerical_features = np.array([
            task_data['Resource Requirements']['CPU'],
            task_data['Resource Requirements']['Memory'],
            task_data['Resource Requirements']['Storage'],
            task_data['Computational Complexity'],
            task_data['Memory Footprint'],
            task_data['Urgency Score'],
            task_data['Expected Value']
        ], dtype=float)

        dependencies_numerical = self.encode_dependencies(task_data['Dependencies'])
        precedence_numerical = self.encode_precedence_relations(task_data.get('Precedence Relations', []))

        combined_embedding = np.concatenate((
            text_embedding,
            type_onehot,
            priority_onehot,
            locality_onehot,
            security_onehot,
            numerical_features,
            dependencies_numerical,
            precedence_numerical
        ))
        print(f"Text Embedding Length: {len(text_embedding)}")
        print(f"Type One-Hot Encoding Length: {len(type_onehot)}")
        print(f"Priority One-Hot Encoding Length: {len(priority_onehot)}")
        print(f"Locality One-Hot Encoding Length: {len(locality_onehot)}")
        print(f"Security One-Hot Encoding Length: {len(security_onehot)}")
        print(f"Numerical Features Length: {len(numerical_features)}")
        print(f"Dependencies Numerical Length: {len(dependencies_numerical)}")
        print(f"Precedence Relations Numerical Length: {len(precedence_numerical)}")
        print(f"Total Combined Embedding Length (Task): {len(combined_embedding)}")
        return combined_embedding

    def one_hot_encode(self, value, categories):
        """
        One-hot encodes a categorical value.
        """
        encoding = np.zeros(len(categories))
        if value in categories:
            encoding[categories.index(value)] = 1
        return encoding

    def encode_dependencies(self, dependencies):
        """
        Encodes task dependencies into a numerical representation.
        """
        max_dependencies = 5
        encoded_dependencies = np.zeros(max_dependencies, dtype=np.float32)

        for i, task_id in enumerate(dependencies):
            if i < max_dependencies:
                task_hash = hash(task_id) % (2**32)
                encoded_dependencies[i] = task_hash

        return encoded_dependencies

    def encode_precedence_relations(self, precedence_relations):
        """
        Encodes precedence relations between tasks into a numerical representation.
        """
        max_relations = 10
        encoded_relations = np.zeros(max_relations * 2, dtype=np.float32)

        for i, relation in enumerate(precedence_relations):
            if i < max_relations:
                try:
                    task_a, task_b = relation.split(" -> ")
                    task_a_hash = hash(task_a) % (2**32)
                    task_b_hash = hash(task_b) % (2**32)
                    encoded_relations[i * 2] = task_a_hash
                    encoded_relations[i * 2 + 1] = task_b_hash
                except ValueError:
                    print(f"Skipping invalid precedence relation format: '{relation}'")

        return encoded_relations

    def encode_graph(self, graph_data):
        """
        Placeholder for encoding graph-level features.
        """
        return np.array([])  # Return an empty array for now

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
    assert len(embedding) == 418

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
    assert len(embedding) == 423

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
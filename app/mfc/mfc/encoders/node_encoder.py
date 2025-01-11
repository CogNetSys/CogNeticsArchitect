# File: mfc/mfc/encoders/node_encoder.py

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch
from collections import OrderedDict

class NodeEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.cache = OrderedDict()
        self.max_cache_size = 10000
        self.max_length = 512

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

        numerical_features = numerical_features / (np.max(numerical_features) if np.any(numerical_features) else 1)

        combined_embedding = np.concatenate((
            text_embedding,
            base_model_onehot,
            role_onehot,
            expertise_onehot,
            np.array(tools_multi_hot),
            numerical_features
        ))

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

        numerical_features = numerical_features / (np.max(numerical_features) if np.any(numerical_features) else 1)

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

        return combined_embedding

    def encode_dependencies(self, dependencies):
        max_dependencies = 5
        encoded_dependencies = np.zeros(max_dependencies, dtype=np.float32)

        for i, task_id in enumerate(dependencies):
            if i < max_dependencies:
                task_hash = hash(task_id) % (2**32)
                encoded_dependencies[i] = task_hash

        return encoded_dependencies

    def encode_precedence_relations(self, precedence_relations):
        max_relations = 10
        encoded_relations = np.zeros(max_relations * 2, dtype=np.float32)

        for i, relation in enumerate(precedence_relations):
            if i < max_relations:
                try:
                    task_a, task_b = relation.split(" -> ")
                    task_a_hash = hash(task_a) % (2**32)
                    task_b_hash = hash(task_b) % (2**32)
                    encoded_relations[i*2] = task_a_hash
                    encoded_relations[i*2 + 1] = task_b_hash
                except ValueError:
                    print(f"Skipping invalid precedence relation format: '{relation}'")

        return encoded_relations

    def one_hot_encode(self, value, categories):
        encoding = np.zeros(len(categories))
        if value in categories:
            encoding[categories.index(value)] = 1
        return encoding

    def encode_graph(self, graph_data):
        # Placeholder for graph encoding logic
        return np.array([])
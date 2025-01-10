# File: mfc/mfc/mfc_manager.py

from typing import Dict, Any, List
import numpy as np
import logging

from mfc.encoder.node_encoder import NodeEncoder
from mfc.modules.feedback_aggregator import FeedbackAggregator
from mfc.modules.deephydra_anomaly_detector import DeepHYDRAAnomalyDetector
from mfc.modules.decision_making import DecisionMakingModule
from mfc.modules.communication import CommunicationModule
from agents.CA_agent import CAAgent

class MFCManager:
    """
    Manages the overall coordination of agents, encoding processes, resource allocation, and communication within the MFC.

    File: mfc/mfc/mfc_manager.py
    """

    def __init__(self, config: Dict[str, Any], llm_api_key: str):
        """
        Initializes the MFC Manager with required components.

        Args:
            config (Dict[str, Any]): Configuration settings for the MFC.
            llm_api_key (str): API key for the language model.
        """
        self.config = config
        self.llm_api_key = llm_api_key
        self.node_encoder = NodeEncoder()
        self.agents: Dict[str, CAAgent] = {}

        # Initialize modules
        self.feedback_aggregator = FeedbackAggregator(anomaly_detector=None)  # To be set after DeepHYDRA is initialized
        self.decision_making = DecisionMakingModule(resource_inventory=config.get("resource_inventory", {}))
        self.communication_module = CommunicationModule(protocol_settings=config.get("protocol_settings", {}))

    def set_anomaly_detector(self, anomaly_detector):
        """
        Sets the anomaly detector for the Feedback Aggregator.

        Args:
            anomaly_detector: An instance of an anomaly detection class.
        """
        self.feedback_aggregator.anomaly_detector = anomaly_detector

    def add_agent(self, agent: CAAgent):
        """
        Adds a CA Agent to the MFC Manager.

        Args:
            agent (CAAgent): The CA Agent to add.
        """
        self.agents[agent.unique_id] = agent
        logging.info(f"Agent {agent.unique_id} added to MFC Manager.")

    def encode_agents(self) -> Dict[str, np.ndarray]:
        """
        Encodes all agents' features.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping agent IDs to their embeddings.
        """
        agent_embeddings = {}
        for agent_id, agent in self.agents.items():
            agent_data = {
                "Base Model": agent.base_model,
                "Role/Expertise": agent.role,
                "Current State": {
                    "Task": agent.current_task,
                    "Resources": agent.resources
                },
                "Available Plugins/Tools": agent.tools,
                "Expertise Level": agent.expertise_level,
                "Current Workload": agent.current_workload,
                "Reliability Score": agent.reliability_score,
                "Latency": agent.latency,
                "Error Rate": agent.error_rate,
                "Cost Per Task": agent.cost_per_task
            }
            embedding = self.node_encoder.encode_agent(agent_data)
            agent_embeddings[agent_id] = embedding
        logging.info("All agent embeddings encoded.")
        return agent_embeddings

    def encode_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Encodes task features.

        Args:
            tasks (List[Dict[str, Any]]): A list of task dictionaries.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping task IDs to their embeddings.
        """
        task_embeddings = {}
        for task in tasks:
            task_data = {
                "Type": task['type'],
                "Resource Requirements": task['resource_requirements'],
                "Deadline": task['deadline'],
                "Dependencies": task['dependencies'],
                "Priority": task['priority'],
                "Computational Complexity": task['computational_complexity'],
                "Memory Footprint": task['memory_footprint'],
                "Data Locality": task['data_locality'],
                "Security Level": task['security_level'],
                "Urgency Score": task['urgency_score'],
                "Expected Value": task['expected_value']
            }
            embedding = self.node_encoder.encode_task(task_data)
            task_embeddings[task['id']] = embedding
        logging.info("All task embeddings encoded.")
        return task_embeddings

    def run_mfc(self, tasks: List[Dict[str, Any]], steps: int):
        """
        Runs the MFC for a specified number of steps, processing tasks and managing resources.

        Args:
            tasks (List[Dict[str, Any]]): A list of task dictionaries to be managed.
            steps (int): Number of simulation steps to run.
        """
        for step in range(steps):
            logging.info(f"--- MFC Step {step + 1} ---")

            # Encode agents and tasks
            agent_embeddings = self.encode_agents()
            task_embeddings = self.encode_tasks(tasks)

            # Aggregate feedback from agents
            feedback_list = [agent.get_feedback() for agent in self.agents.values()]
            aggregated_feedback = self.feedback_aggregator.aggregate_feedback(feedback_list)

            # Prioritize tasks
            prioritized_tasks = self.decision_making.prioritize_tasks(task_embeddings, agent_embeddings)

            # Allocate resources
            allocation = self.decision_making.allocate_resources(prioritized_tasks, agent_embeddings)

            # Communicate allocations to agents
            for task_id, resources in allocation.items():
                message = {
                    "task_id": task_id,
                    "allocated_resources": resources
                }
                # For simplicity, assign to the first available agent
                if self.agents:
                    recipient_id = next(iter(self.agents))
                    self.communication_module.send_message(recipient_id, message)

            # Update agents
            for agent in self.agents.values():
                neighbors = agent.get_neighbors()
                agent.step(neighbors)

            logging.info(f"Step {step + 1} completed.")
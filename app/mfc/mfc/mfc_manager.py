# File: mfc/mfc/mfc_manager.py

import logging
from typing import Dict, Any, List
import numpy as np

from mfc.agents.elicitor_agent import ElicitorAgent
from mfc.agents.rule_agent import RuleAgent
from mfc.agents.rule_distributor_agent import RuleDistributorAgent
from mfc.mfc.modules.feedback_aggregator import FeedbackAggregator
from mfc.mfc.modules.deephydra_anomaly_detector import DeepHYDRAAnomalyDetector
from mfc.mfc.modules.decision_making import DecisionMakingModule
from mfc.mfc.modules.communication import CommunicationModule
from mfc.agents.CA_agent import CAAgent
from mfc.mfc.encoders.node_encoder import NodeEncoder

class MFCManager:
    def __init__(self, config: Dict[str, Any], llm_api_key: str,
                 feedback_aggregator: FeedbackAggregator,
                 decision_making: DecisionMakingModule,
                 communication_module: CommunicationModule,
                 node_encoder: NodeEncoder):
        """
        Initializes the MFC Manager with configurations and components.

        Args:
            config (dict): Configuration settings for the MFC.
            llm_api_key (str): API key for the LLM.
            feedback_aggregator (FeedbackAggregator): Instance for aggregating feedback.
            decision_making (DecisionMakingModule): Instance for decision-making.
            communication_module (CommunicationModule): Instance for handling communication.
            node_encoder (NodeEncoder): Instance for encoding features.
        """
        self.config = config
        self.llm_api_key = llm_api_key
        self.agents: Dict[str, CAAgent] = {}
        self.feedback_aggregator = feedback_aggregator
        self.decision_making = decision_making
        self.communication_module = communication_module
        self.node_encoder = node_encoder

        # Initialize other agents
        self.elicitor_agent = ElicitorAgent(llm_api_key)
        self.rule_agent = RuleAgent(llm_api_key)

        # Initialize RuleDistributorAgent after agents have been added
        self.rule_distributor_agent = RuleDistributorAgent(ca_registry=self.agents)

        logging.info("MFC Manager initialized.")

    def add_agent(self, agent: CAAgent):
        """Adds an agent to the MFC."""
        self.agents[agent.unique_id] = agent
        logging.info(f"Added agent: {agent.unique_id}")
        # Optionally, update RuleDistributorAgent's registry if needed
        self.rule_distributor_agent.ca_registry = self.agents

    def remove_agent(self, agent_id: str):
        """Removes an agent from the MFC."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logging.info(f"Removed agent: {agent_id}")
            # Optionally, update RuleDistributorAgent's registry if needed
            self.rule_distributor_agent.ca_registry = self.agents
        else:
            logging.warning(f"Agent {agent_id} not found.")

    def update_goals(self, new_goals):
        """Updates the system goals based on new inputs."""
        logging.info("Updating system goals.")
        # Implement logic to update goals based on new inputs

    def distribute_rules(self):
        """Distributes rules to relevant CAs."""
        logging.info("Distributing rules to CAs.")
        self.rule_distributor_agent.distribute_rules()

    def collect_feedback(self) -> Dict[str, Any]:
        """
        Collects feedback from all agents.

        Returns:
            Dict[str, Any]: Combined feedback from all agents.
        """
        feedback_list = [agent.get_feedback() for agent in self.agents.values()]
        logging.info("Collecting feedback from agents.")
        return self.feedback_aggregator.aggregate_feedback(feedback_list)

    def analyze_patterns(self, feedback: Dict[str, Any]):
        """
        Analyzes the collected feedback for patterns.

        Args:
            feedback (Dict[str, Any]): The aggregated feedback data.
        """
        logging.info("Analyzing patterns in feedback.")
        # Implement pattern analysis logic here
        pass

    def adjust_resource_allocation(self):
        """
        Adjusts resource allocation based on the detected patterns.
        """
        logging.info("Adjusting resource allocation.")
        # Implement resource allocation adjustment logic here

    def encode_agents(self) -> Dict[str, np.ndarray]:
        """
        Encodes all agents' features using the NodeEncoder.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping agent IDs to their embeddings.
        """
        agent_embeddings = {}
        for agent_id, agent in self.agents.items():
            agent_data = {
                "Base Model": agent.base_model,
                "Role/Expertise": agent.role,
                "Current State": {
                    "Task": agent.state,  # Assuming 'state' represents 'Task'
                    "Resources": agent.resource
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
        logging.info("Encoded all agent embeddings.")
        return agent_embeddings

    def encode_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Encodes task features using the NodeEncoder.

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
                "Expected Value": task['expected_value'],
                "Precedence Relations": task.get('precedence_relations', [])
            }
            embedding = self.node_encoder.encode_task(task_data)
            task_embeddings[task['id']] = embedding
        logging.info("Encoded all task embeddings.")
        return task_embeddings

    def is_neighbor(self, agent_id1: str, agent_id2: str) -> bool:
        """
        Determines if two agents are neighbors based on their IDs.

        Args:
            agent_id1 (str): Unique ID of the first agent.
            agent_id2 (str): Unique ID of the second agent.

        Returns:
            bool: True if agents are neighbors, False otherwise.
        """
        # Example logic: Sequential agents are neighbors
        if agent_id1 == 'CA1' and agent_id2 == 'CA2':
            return True
        if agent_id1 == 'CA2' and agent_id2 == 'CA3':
            return True
        return False
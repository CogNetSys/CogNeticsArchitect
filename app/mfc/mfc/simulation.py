# mfc/mfc/simulation.py

import os
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional
from goal_pattern_detector.goal_pattern_detector import GoalPatternDetector
from goal_pattern_detector.context_embedding import ContextEmbedding
from goal_pattern_detector.time_window_manager import TimeWindowManager
from data_generation import load_time_series_data
from collections import defaultdict
from agents.CA_agent import CAAgent
from mfc_manager import MFCManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CAAgent:
    def __init__(self, unique_id: str, model, initial_state: int = 0, initial_resource: int = 5, message_queue: Optional[List[str]] = None, behavior: Optional[str] = None):
        """
        Initializes a CA Agent with specified behaviors.

        Args:
            unique_id (str): Unique identifier for the CA.
            model: The MFC model this agent is part of.
            initial_state (int): Initial state of the CA.
            initial_resource (int): Initial resource level of the CA.
            message_queue (Optional[list]): A queue to store messages for the agent.
            behavior (str): The behavior type of the CA (e.g., 'oscillatory', 'dependent', 'sudden_change').
        """
        self.unique_id = unique_id
        self.model = model
        self.state = initial_state
        self.resource = initial_resource
        self.message_queue = message_queue or []
        self.behavior = behavior or 'default'
        self.behavior_params = {'noise_std': 0.5}  # Example parameter; can be extended

    def step(self, neighbors: List['CAAgent']):
        """
        Executes a simulation step for the CA Agent.

        Args:
            neighbors (List[CAAgent]): List of neighboring CA agents.
        """
        # Process incoming messages
        while self.message_queue:
            message = self.message_queue.pop(0)
            if message == "Increase":
                self.state += 1
                logging.debug(f"{self.unique_id} received 'Increase' message and updated state to {self.state}")

        # Execute behavior based on type
        if self.behavior == 'oscillatory':
            self.oscillatory_behavior()
        elif self.behavior == 'dependent':
            self.dependent_behavior(neighbors)
        elif self.behavior == 'sudden_change':
            self.sudden_change_behavior()
        else:
            self.default_behavior()

        # Ensure state stays within bounds [0, 10]
        self.state = max(0, min(self.state, 10))

        # Resource management
        resource_change = random.choice([-2, -1, 0, 1, 2])
        self.resource = max(0, min(self.resource + resource_change, 10))
        logging.debug(f"{self.unique_id} resource level: {self.resource}")

        # Send messages based on current state
        if self.state > 5 and neighbors:
            neighbor = random.choice(neighbors)
            self.send_message(neighbor, "Increase")

    def oscillatory_behavior(self):
        """
        Defines oscillatory behavior for the CA Agent.
        """
        oscillate_pattern = [1, 2, 3, 2, 1]
        self.state = oscillate_pattern[self.state % len(oscillate_pattern)]
        logging.debug(f"{self.unique_id} performed oscillatory behavior. New state: {self.state}")

    def dependent_behavior(self, neighbors: List['CAAgent']):
        """
        Defines dependent behavior where the agent reacts to neighbors.

        Args:
            neighbors (List[CAAgent]): List of neighboring CA agents.
        """
        if neighbors:
            # Example: React to the first neighbor's state
            neighbor = neighbors[0]
            if neighbor.state > 2:
                self.state += 1
                logging.debug(f"{self.unique_id} reacts to {neighbor.unique_id}'s state. New state: {self.state}")

    def sudden_change_behavior(self):
        """
        Defines sudden change behavior for the CA Agent.
        """
        if random.random() < 0.3:
            self.state += 2  # Sudden jump
            logging.warning(f"{self.unique_id} experienced a sudden change. New state: {self.state}")
        else:
            self.state -= 1  # Gradual decrease

    def default_behavior(self):
        """
        Defines default behavior for the CA Agent.
        """
        if random.random() < 0.5:
            self.state += 1
        else:
            self.state -= 1
        logging.debug(f"{self.unique_id} performed default behavior. New state: {self.state}")

    def send_message(self, neighbor: 'CAAgent', message: str):
        """
        Sends a message to a neighboring CA Agent.

        Args:
            neighbor (CAAgent): The neighboring CA Agent to send the message to.
            message (str): The message content.
        """
        neighbor.receive_message(message)
        logging.debug(f"{self.unique_id} sent message '{message}' to {neighbor.unique_id}")

    def receive_message(self, message: str):
        """
        Receives a message from another CA Agent.

        Args:
            message (str): The message content.
        """
        self.message_queue.append(message)
        logging.debug(f"{self.unique_id} received message '{message}'.")


class MFCManager:
    def __init__(self, config: Dict, context_embedding: ContextEmbedding, time_window_manager: TimeWindowManager, llm_api_key: str):
        """
        Initializes the MFCManager.

        Args:
            config (dict): Configuration parameters for the MFC.
            context_embedding (ContextEmbedding): Instance of ContextEmbedding.
            time_window_manager (TimeWindowManager): Instance of TimeWindowManager.
            llm_api_key (str): API key for the LLM (e.g., OpenAI).
        """
        self.agents: Dict[str, CAAgent] = {}
        self.context_embedding = context_embedding
        self.time_window_manager = time_window_manager
        self.goal_pattern_detector = GoalPatternDetector(
            significance_threshold=0.05,
            min_pattern_length=3,
            initial_threshold=0.9,
            min_threshold=0.6,
            adaptation_rate=0.05,
            fixed_window_size=100,
            max_windows=100,
            llm_api_key=llm_api_key
        )
        self.metrics = {
            'execution_time': [],
            'memory_usage': [],
            'patterns_detected': [],
            'unique_patterns': [],
            'rules_generated': [],
            'errors': []
        }
        logging.info("Initialized MFCManager.")

    def add_agent(self, agent: CAAgent):
        """
        Adds a CA Agent to the MFC.

        Args:
            agent (CAAgent): The CA Agent to add.
        """
        self.agents[agent.unique_id] = agent
        logging.info(f"Added agent {agent.unique_id} with behavior {agent.behavior}.")

    def run_mfc(self, steps: int = 1000, prune_interval: int = 100, min_weight: int = 5):
        """
        Runs the MFC simulation for a specified number of steps.

        Args:
            steps (int): Number of simulation steps to run.
            prune_interval (int): Interval at which to prune the pattern graph.
            min_weight (int): Minimum weight threshold for pruning.
        """
        for step in range(steps):
            logging.info(f"--- Simulation Step {step+1} ---")
            try:
                # Update all agents
                for agent in self.agents.values():
                    neighbors = self.get_neighbors(agent.unique_id)
                    agent.step(neighbors)

                # Collect data for pattern detection
                data = {ca_id: agent.state for ca_id, agent in self.agents.items()}

                # Record transitions and contexts
                for ca_id, state in data.items():
                    # Example: Dummy previous state; replace with actual tracking
                    old_state = {'state': state - 1} if state > 0 else {'state': 0}
                    new_state = {'state': state}
                    context = {
                        'ca_states': data,
                        'resources': {ca_id: agent.resource for ca_id, agent in self.agents.items()},
                        'active_goals': [],  # Extend as needed
                        'environment': {}  # Extend as needed
                    }
                    self.goal_pattern_detector.record_state_transition(
                        ca_id=ca_id,
                        old_state=old_state,
                        new_state=new_state,
                        context=context
                    )

                # Detect patterns
                patterns = self.goal_pattern_detector.detect_temporal_patterns(data, adaptive=True)
                self.metrics['patterns_detected'].append(len(patterns))
                self.metrics['unique_patterns'].append(len(self.goal_pattern_detector.patterns))
                self.metrics['rules_generated'].append(len(self.goal_pattern_detector.rules))

                # Log patterns
                for pattern in patterns:
                    logging.info(f"Detected Pattern: {pattern}")

                # Act on detected patterns and generated rules
                for pattern in patterns:
                    rule = pattern['rule']
                    # Example action: Execute the rule's action
                    self.execute_rule(rule)

                # Prune the pattern graph at defined intervals
                if (step + 1) % prune_interval == 0:
                    self.goal_pattern_detector.prune_pattern_graph(min_weight=min_weight)

            except Exception as e:
                logging.error(f"Error during simulation step {step+1}: {e}")
                self.metrics['errors'].append(str(e))

            # TODO: Collect and record additional metrics as needed

    def get_neighbors(self, ca_id: str) -> List[CAAgent]:
        """
        Retrieves neighboring CA Agents for a given CA Agent.

        Args:
            ca_id (str): The ID of the CA Agent.

        Returns:
            list: List of neighboring CA Agents.
        """
        # Placeholder for neighbor retrieval logic
        # For simplicity, return all other agents as neighbors
        return [agent for id_, agent in self.agents.items() if id_ != ca_id]

    def execute_rule(self, rule: str):
        """
        Executes the action specified in the IF-THEN rule.

        Args:
            rule (str): The IF-THEN rule to execute.
        """
        if rule == "IF [conditions] THEN [actions].":
            logging.warning("Received default fallback rule. No action taken.")
            return

        # Simple parser to extract actions
        try:
            actions_part = rule.split("then")[1].strip().strip('.')
            actions = actions_part.split(" and ")
            for action in actions:
                if "increase resource allocation" in action:
                    self.increase_resources()
                elif "trigger an alert" in action:
                    self.trigger_alert()
                # Add more action handlers as needed
            logging.info(f"Executed actions from rule: {rule}")
        except IndexError:
            logging.error(f"Rule parsing failed for rule: {rule}")

    def increase_resources(self):
        """
        Example action to increase resource allocation.
        """
        for agent in self.agents.values():
            agent.resource = min(agent.resource + 1, 10)
            logging.debug(f"Increased resources for {agent.unique_id} to {agent.resource}")

    def trigger_alert(self):
        """
        Example action to trigger an alert.
        """
        logging.warning("Alert triggered by GoalPatternDetector!")

if __name__ == "__main__":
    # Load external time series data
    external_data = load_time_series_data("path_to_time_series_dataset.csv")

    # Initialize components
    context_embedding = ContextEmbedding()
    time_window_manager = TimeWindowManager(fixed_size=100, overlap_ratio=0.5, max_lag=5)
    llm_api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment
    if llm_api_key is None:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    # Initialize MFC Manager
    mfc_model = MFCManager(
        config={},  # Add appropriate configuration if needed
        context_embedding=context_embedding,
        time_window_manager=time_window_manager,
        llm_api_key=llm_api_key
    )

    # Create and add CAAgents with diverse behaviors
    for ca_id, data_sequence in external_data.items():
        behavior_type = random.choice(['oscillatory', 'dependent', 'sudden_change', 'default'])
        agent = CAAgent(unique_id=ca_id, model=mfc_model, initial_state=0, behavior=behavior_type)
        mfc_model.add_agent(agent)

    # Run Simulation for 50 steps
    mfc_model.run_mfc(steps=50)

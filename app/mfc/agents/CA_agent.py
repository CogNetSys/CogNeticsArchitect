# File: app/mfc/agents/CA_agent.py

import random
import logging
from typing import Any, Dict

class CAAgent:
    def __init__(self, unique_id, model, initial_state=0, initial_resource=5, message_queue=None, behavior=None,
                 base_model=None, role=None, tools=None, expertise_level=None, current_workload=None,
                 reliability_score=None, latency=None, error_rate=None, cost_per_task=None):
        """
        Initializes a CA Agent with specified behaviors.

        Args:
            unique_id (str): Unique identifier for the CA.
            model (MFCManager): The MFC model this agent is part of.
            initial_state (int): Initial state of the CA.
            initial_resource (int): Initial resource level of the CA.
            message_queue (Optional[list]): A queue to store messages for the agent.
            behavior (str): The behavior type of the CA (e.g., 'oscillatory', 'dependent', 'sudden_change').
            base_model (str): The foundational language model used by the agent.
            role (str): The specialized function or expertise area of the agent.
            tools (list): Tools that the agent can utilize to perform tasks.
            expertise_level (int): The level of expertise of the agent.
            current_workload (int): The current workload of the agent.
            reliability_score (float): The reliability score of the agent.
            latency (float): Average response time of the agent.
            error_rate (float): Frequency of errors or failures in task execution.
            cost_per_task (float): Cost associated with using the agent.
        """
        self.unique_id = unique_id
        self.model = model
        self.state = initial_state
        self.resource = initial_resource
        self.message_queue = message_queue or []
        self.behavior = behavior or 'default'
        self.behavior_params = {'noise_std': 0.5}  # Example parameter; can be extended
        self.base_model = base_model
        self.role = role
        self.tools = tools
        self.expertise_level = expertise_level
        self.current_workload = current_workload
        self.reliability_score = reliability_score
        self.latency = latency
        self.error_rate = error_rate
        self.cost_per_task = cost_per_task
        self.rules = []  # Initialize an empty list to store rules

    def step(self, neighbors):
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

        # Apply rules
        self.apply_rules()

        # Ensure state stays within bounds [0, 10]
        self.state = max(0, min(self.state, 10))

        # Resource management
        resource_change = random.choice([-2, -1, 0, 1, 2])
        self.resource = max(0, min(self.resource + resource_change, 10))
        logging.debug(f"{self.unique_id} resource level: {self.resource}")

        # Send messages based on current state
        if self.state > 5 and neighbors:
            # Randomly select a neighbor to send a message to
            neighbor = random.choice(neighbors)
            self.send_message(neighbor, "Increase")

    def oscillatory_behavior(self):
        """
        Defines oscillatory behavior for the CA Agent.
        """
        oscillate_pattern = [1, 2, 3, 2, 1]
        self.state = oscillate_pattern[self.state % len(oscillate_pattern)]
        logging.debug(f"{self.unique_id} performed oscillatory behavior. New state: {self.state}")

    def dependent_behavior(self, neighbors):
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

    def apply_rules(self):
        """
        Applies assigned rules to the CA Agent.
        """
        for rule in self.rules:
            # Example: Apply rule based on task type
            task_type = rule.get("task_type")
            action = rule.get("action")

            if task_type == "Development" and action == "Increase workload":
                self.current_workload += 1
                logging.info(f"{self.unique_id} increased workload to {self.current_workload} based on rule.")
            elif task_type == "Data Analysis" and action == "Optimize resources":
                self.resource = max(1, self.resource - 1)
                logging.info(f"{self.unique_id} optimized resources to {self.resource} based on rule.")
            elif task_type == "Mathematical Computation" and action == "Allocate more memory":
                self.resource += 2
                logging.info(f"{self.unique_id} allocated more memory to {self.resource} based on rule.")
            # Add more rule conditions as needed

    def send_message(self, neighbor, message):
        """
        Sends a message to a neighboring CA Agent.

        Args:
            neighbor (CAAgent): The neighboring CA Agent to send the message to.
            message (str): The message content.
        """
        neighbor.receive_message(message)
        logging.debug(f"{self.unique_id} sent message '{message}' to {neighbor.unique_id}")

    def receive_message(self, message):
        """
        Receives a message from another CA Agent.

        Args:
            message (str): The message content.
        """
        self.message_queue.append(message)
        logging.debug(f"{self.unique_id} received message '{message}'.")

    def get_neighbors(self):
        """
        Returns the neighboring agents of the current agent.

        Returns:
            list: A list of neighboring CAAgent instances.
        """
        neighbors = []
        for agent in self.model.agents.values():
            if agent != self and self.model.is_neighbor(self.unique_id, agent.unique_id):
                neighbors.append(agent)
        return neighbors

    def is_neighbor(self, other_agent_id):
        """
        Checks if another agent is a neighbor based on a simple proximity rule.

        Args:
            other_agent_id (str): Unique ID of the other agent.

        Returns:
            bool: True if the other agent is a neighbor, False otherwise.
        """
        # Example: Consider CA2 as a neighbor of CA1, and CA3 as a neighbor of CA2
        if self.unique_id == 'CA1' and other_agent_id == 'CA2':
            return True
        if self.unique_id == 'CA2' and other_agent_id == 'CA3':
            return True
        return False

    def get_feedback(self):
        """
        Provides feedback on the agent's performance.

        Returns:
            dict: Feedback data.
        """
        # Placeholder for feedback mechanism
        return {
            "unique_id": self.unique_id,
            "state": self.state,
            "resource": self.resource
        }

    def receive_rule(self, rule: Dict[str, Any]):
        """
        Receives a rule from the RuleDistributorAgent.

        Args:
            rule (Dict[str, Any]): The rule to be applied.
        """
        self.rules.append(rule)
        logging.info(f"{self.unique_id} received rule: {rule}")

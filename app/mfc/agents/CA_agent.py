# mfc/agents/ca_agent.py
import random
import logging

class CAAgent:
    def __init__(self, unique_id, model, initial_state=0, initial_resource=5, message_queue=None):
        """
        Initializes a CA Agent.

        Args:
            unique_id (str): Unique identifier for the CA.
            model (MFCModel): The MFC model this agent is part of.
            initial_state (int): Initial state of the CA.
            initial_resource (int): Initial resource level of the CA.
            message_queue (Optional[list]): A queue to store messages for the agent.
        """
        self.unique_id = unique_id
        self.model = model
        self.state = initial_state
        self.resource = initial_resource
        self.message_queue = message_queue or []

    def send_message(self, target_agent, message: str):
        """Sends a message to a target agent."""
        if target_agent is not None:
            target_agent.message_queue.append(message)
            logging.debug(f"{self.unique_id} sent message to {target_agent.unique_id}: {message}")

    def step(self, neighbors):
        """
        Defines the state transition logic for the CA, considering messages from neighbors.
        """
        # Process incoming messages
        while self.message_queue:
            message = self.message_queue.pop(0)
            if message == "Increase":
                self.state += 1
                logging.debug(f"{self.unique_id} received 'Increase' message and updated state to {self.state}")

        # State transition logic based on CA ID
        if self.unique_id == 'CA1':
            # CA1 increments its state with a 70% probability
            if random.random() < 0.7:
                self.state += 1
        elif self.unique_id == 'CA2':
            # CA2 decrements its state with a 50% probability
            if random.random() < 0.5:
                self.state -= 1
        elif self.unique_id == 'CA3':
            # CA3 toggles its state between 0 and 1
            self.state = 1 - self.state

        # Ensure state stays within bounds [0, 10]
        self.state = max(0, min(self.state, 10))

        # Resource management
        resource_change = random.choice([-1, 0, 1])
        self.resource = max(0, min(self.resource + resource_change, 10))

        logging.debug(f"{self.unique_id} state updated to {self.state}, resource level: {self.resource}")

        # Send messages based on current state
        if self.state > 5 and neighbors:
            # Randomly select a neighbor to send a message to
            neighbor = random.choice(neighbors)
            self.send_message(neighbor, "Increase")

    def get_neighbors(self):
        """
        Returns the neighboring agents of the current agent.

        Returns:
            list: A list of neighboring CAAgent instances.
        """
        neighbors = []
        for agent in self.model.schedule.agents:
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
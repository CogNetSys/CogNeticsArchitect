# File: app/mfc/agents/rule_distributor_agent.py

import logging
from typing import Dict, Any

from mfc.agents.CA_agent import CAAgent

class RuleDistributorAgent:
    def __init__(self, ca_registry: Dict[str, 'CAAgent']):
        """
        Initializes the RuleDistributorAgent with a registry of CA agents.

        Args:
            ca_registry (Dict[str, CAAgent]): A dictionary of CA agents.
        """
        self.ca_registry = ca_registry
        logging.info("RuleDistributorAgent initialized with CA registry.")

    def distribute_rules(self):
        """
        Distributes rules to the relevant CA agents.
        """
        logging.info("Distributing rules to CAs.")
        for agent_id, agent in self.ca_registry.items():
            if agent.role == "Programmer":
                rule = {"task_type": "Development", "action": "Increase workload"}
                agent.receive_rule(rule)
                logging.info(f"Assigned rule to {agent_id}: {rule}")
            elif agent.role == "Data Analyst":
                rule = {"task_type": "Data Analysis", "action": "Optimize resources"}
                agent.receive_rule(rule)
                logging.info(f"Assigned rule to {agent_id}: {rule}")
            elif agent.role == "Mathematician":
                rule = {"task_type": "Mathematical Computation", "action": "Allocate more memory"}
                agent.receive_rule(rule)
                logging.info(f"Assigned rule to {agent_id}: {rule}")
            # Add more conditions as needed

# File: app/mfc/agents/rule_agent.py

import json
import logging
from typing import Dict, Any
from smolagents import CodeAgent

class RuleAgent:
    def __init__(self, llm_api_key: str, initial_rules: list = None):
        """
        Initializes the RuleAgent with specified rules.

        Args:
            llm_api_key (str): API key for the language model.
            initial_rules (list): List of initial rules for the agent.
        """
        self.llm_api_key = llm_api_key
        self.initial_rules = initial_rules or []
        # Initialize other attributes as needed

    def classify_goal(self, goal_text: str) -> str:
        """
        Classifies the goal text into a specific category.

        Args:
            goal_text (str): The text describing the goal.

        Returns:
            str: The classified goal category.
        """
        if "increase customer retention" in goal_text.lower():
            return "increase_customer_retention"
        elif "reduce churn" in goal_text.lower():
            return "reduce_churn"
        elif "reduce cost" in goal_text.lower():
            return "reduce_cost"
        else:
            return "unknown_goal"

    def generate_rule(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a rule based on the structured data.

        Args:
            structured_data (Dict[str, Any]): The structured data containing goal, metric, constraint, timeline.

        Returns:
            Dict[str, Any]: The generated rule data.
        """
        goal_text = structured_data.get("goal", "Unknown goal")
        classified_goal = self.classify_goal(goal_text)

        if classified_goal not in self.rule_templates:
            raise ValueError(f"Invalid goal: {goal_text}. No rule available for this type of goal.")

        rule = self.rule_templates[classified_goal]
        rule_data = {
            "rule": rule,
            "class": classified_goal,
            "context": structured_data.get("context", "general"),
        }

        logging.info(f"Generated Rule: {rule}")
        logging.info(f"Rule Data for Distribution: {rule_data}")
        return rule_data

    def process_nn_output(self, predicted_class: int) -> Dict[str, Any]:
        """
        Generate a rule based on the demand class index.

        Args:
            predicted_class (int): The index predicted by the NN for "low", "normal", or "high".

        Returns:
            dict: The generated rule data for distribution.
        """
        # Class index-to-label mapping
        demand_classes = ["low", "normal", "high"]

        if predicted_class not in [0, 1, 2]:
            raise ValueError(f"Invalid predicted class index: {predicted_class}")

        demand_label = demand_classes[predicted_class]  # Convert class index to label
        logging.info(f"Demand Class Received from NN: {demand_label}")

        # Generate a rule based on the NN output
        rule = self.rule_templates[demand_label]

        # Prepare rule data for distribution
        rule_data = {
            "rule": rule,
            "class": demand_label,
            "context": "onboarding process"
        }

        logging.info(f"Generated Rule: {rule_data['rule']}")
        logging.info(f"Rule Data for Distribution: {json.dumps(rule_data, indent=4)}")

        return rule_data

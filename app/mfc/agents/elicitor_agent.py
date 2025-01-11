# File: app/mfc/agents/elicitor_agent.py

import json
import logging
from typing import Dict, Any
from smolagents import CodeAgent

class ElicitorAgent:
    def __init__(self, llm_api_key: str, initial_rules: list = None):
        """
        Initializes the ElicitorAgent with specified rules.

        Args:
            llm_api_key (str): API key for the language model.
            initial_rules (list): List of initial rules for the agent.
        """
        self.llm_api_key = llm_api_key
        self.initial_rules = initial_rules or []
        logging.info("ElicitorAgent initialized with initial rules.")
        # Initialize other attributes as needed

    def receive_rule(self, rule: Dict[str, Any]):
        """
        Receives a rule and processes it.

        Args:
            rule (Dict[str, Any]): The rule to be processed.
        """
        # Implement rule processing logic here
        logging.info(f"Received rule: {rule}")
        self.initial_rules.append(rule)

    def extract_structured_data(self, input_text: str) -> Dict[str, Any]:
        """
        Extracts structured data from the input text.

        Args:
            input_text (str): The unstructured input text.

        Returns:
            Dict[str, Any]: The extracted structured data.
        """
        try:
            # Use DataExtractorTool via code_agent to extract structured data
            prompt = "Use the data_extractor tool to extract structured data from the input text."
            additional_args = {"input_text": input_text}
            extraction_result = self.code_agent.run(prompt, additional_args=additional_args)

            # Debug: Print the raw extraction result
            logging.debug(f"Raw Extraction Result: {extraction_result}")

            # Ensure extraction result is a valid JSON string or dict
            if isinstance(extraction_result, str):
                try:
                    structured_data = json.loads(extraction_result.replace("'", '"'))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in extraction result: {e}\nResult: {extraction_result}")
            elif isinstance(extraction_result, dict):
                structured_data = extraction_result
            else:
                raise ValueError(f"Unexpected extraction result type: {type(extraction_result)}")

            # Check if structured data is valid
            if not isinstance(structured_data, dict):
                raise ValueError(f"Structured data is not a dictionary: {structured_data}")

            logging.info(f"Extracted Structured Data: {structured_data}")

            # Clarification for missing fields
            for key, value in structured_data.items():
                if not value or value.lower() == "n/a":
                    clarification = self.clarify_missing_field(key, input_text)
                    structured_data[key] = clarification

            logging.info(f"Final Structured Data: {structured_data}")

            return structured_data

        except Exception as e:
            logging.error(f"Error during structured data elicitation: {e}")
            raise

    def clarify_missing_field(self, missing_key: str, input_text: str) -> str:
        """
        Clarifies missing fields using the ClarificationTool via code_agent.

        Args:
            missing_key (str): The missing element to clarify (e.g., goal, metric).
            input_text (str): The original directive or feedback text.

        Returns:
            str: The clarified value.
        """
        try:
            prompt = "Use the clarification_agent tool to ask for missing information."
            additional_args = {"missing_key": missing_key, "input_text": input_text}
            clarification = self.code_agent.run(prompt, additional_args=additional_args)

            logging.info(f"Clarified {missing_key}: {clarification}")
            return clarification

        except Exception as e:
            logging.error(f"Error during clarification of {missing_key}: {e}")
            return "N/A"

# File: mfc/mfc/modules/communication.py

import json
import logging
from typing import Dict, Any

class CommunicationModule:
    """
    Handles communication between the MFC and Cellular Automata (CAs) using the AC2C protocol.

    File: mfc/mfc/modules/communication.py
    """

    def __init__(self, protocol_settings: Dict[str, Any]):
        """
        Initializes the CommunicationModule with protocol settings.

        Args:
            protocol_settings (Dict[str, Any]): Configuration settings for the communication protocol.
        """
        self.protocol_settings = protocol_settings

    def send_message(self, recipient_id: str, message: Dict[str, Any]):
        """
        Sends a message to a specific CA.

        Args:
            recipient_id (str): The unique identifier of the recipient CA.
            message (Dict[str, Any]): The message content.
        """
        # Serialize message to JSON
        message_json = json.dumps(message)
        # Placeholder: Implement actual sending logic (e.g., via sockets, HTTP, etc.)
        logging.info(f"Sending message to {recipient_id}: {message_json}")
        # Example: print(message_json)
        pass

    def receive_message(self, message_json: str) -> Dict[str, Any]:
        """
        Receives and deserializes a message from a CA.

        Args:
            message_json (str): The JSON-formatted message.

        Returns:
            Dict[str, Any]: The deserialized message content.
        """
        try:
            message = json.loads(message_json)
            logging.info(f"Received message: {message}")
            return message
        except json.JSONDecodeError:
            logging.error("Failed to decode incoming message.")
            return {}

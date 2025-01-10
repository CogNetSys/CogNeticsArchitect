# File: mfc/mfc/modules/feedback_aggregator.py

from typing import List, Dict, Any
import numpy as np
import logging
# from .deephydra_anomaly_detector import DeepHYDRAAnomalyDetector

class FeedbackAggregator:
    """
    Aggregates feedback from multiple Cellular Automata (CAs) and detects anomalies.
    """

    def __init__(self, anomaly_detector=None):
        """
        Initializes the FeedbackAggregator with an anomaly detection component.

        Args:
            anomaly_detector: An instance of an anomaly detection class (e.g., DeepHYDRA).
        """
        self.anomaly_detector = anomaly_detector

    def aggregate_feedback(self, feedback_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates feedback from multiple CAs.

        Args:
            feedback_list (List[Dict[str, Any]]): A list of feedback dictionaries from CAs.

        Returns:
            Dict[str, Any]: Aggregated feedback data.
        """
        aggregated = {
            "average_state": 0.0,
            "average_resource": 0.0,
            "anomalies": []
        }

        if not feedback_list:
            logging.warning("No feedback received to aggregate.")
            return aggregated

        states = [fb.get("state", 0) for fb in feedback_list]
        resources = [fb.get("resource", 0) for fb in feedback_list]

        aggregated["average_state"] = np.mean(states)
        aggregated["average_resource"] = np.mean(resources)

        # Detect anomalies using the provided anomaly detector
        if self.anomaly_detector:
            anomalies = self.anomaly_detector.detect_anomalies(feedback_list)
            aggregated["anomalies"] = anomalies
        else:
            logging.warning("Anomaly detector not initialized.")

        logging.info(f"Aggregated Feedback: {aggregated}")
        return aggregated
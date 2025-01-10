# File: mfc/mfc/modules/deephydra_anomaly_detector.py

from typing import List, Dict, Any

class DeepHYDRAAnomalyDetector:
    """
    Integrates DeepHYDRA for anomaly detection in aggregated feedback.

    File: mfc/mfc/modules/deephydra_anomaly_detector.py
    """

    def __init__(self, model):
        """
        Initializes the anomaly detector with a trained DeepHYDRA model.

        Args:
            model: A pre-trained DeepHYDRA model instance.
        """
        self.model = model

    def detect_anomalies(self, feedback_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects anomalies in the feedback data using DeepHYDRA.

        Args:
            feedback_list (List[Dict[str, Any]]): A list of feedback dictionaries from CAs.

        Returns:
            List[Dict[str, Any]]: A list of detected anomalies.
        """
        # Convert feedback_list to the format expected by DeepHYDRA
        data = [fb["state"] for fb in feedback_list]
        predictions = self.model.predict(data)

        anomalies = []
        for fb, pred in zip(feedback_list, predictions):
            if pred == 1:  # Assuming 1 indicates anomaly
                anomalies.append({
                    "agent_id": fb.get("unique_id", "Unknown"),
                    "issue": "Detected anomaly in state or resource utilization."
                })

        return anomalies

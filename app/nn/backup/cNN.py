import numpy as np
import tensorflow as tf  # Neural network import
from app.agents.RuleAgent import (
    process_nn_output,
)  # Import function to process the demand class
import json

# Sample neural network model initialization
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(
            3, activation="softmax"
        ),  # Output layer for 'low', 'normal', 'high'
    ]
)


def process_structured_data(structured_data):
    """
    Process the structured data through the neural network and classify the demand.

    Args:
        structured_data (dict): A dictionary containing 'goal', 'metric', 'constraint', and 'timeline'.
    """
    try:
        # Extract fields
        goal = structured_data.get("goal", "unknown")
        constraint = structured_data.get("constraint", "unknown")
        timeline = structured_data.get("timeline", "unknown")

        # Handle metric conversion
        metric_str = structured_data.get("metric", "0")
        metric_value = (
            int(metric_str.replace("%", "").strip()) if metric_str else 0
        )  # Convert metric to int

        # Prepare input features for NN
        input_features = np.array(
            [[len(goal), metric_value, len(constraint), len(timeline)]]
        )
        print(f"Input Features for NN: {input_features}")

        # Run through NN
        prediction = model.predict(input_features)
        demand_class_index = int(
            np.argmax(prediction)
        )  # Get class index from NN output

        # Call process_nn_output to generate rule and print it
        process_nn_output(demand_class_index)
    except ValueError as ve:
        print(f"Value Error: {ve}. Please check the input values.")
    except Exception as e:
        print(f"Error processing data: {e}")


# Example usage if testing directly
if __name__ == "__main__":
    sample_data = {
        "goal": "increase customer retention",
        "metric": "15%",
        "constraint": "Budget cap: $500K",
        "timeline": "Q1 2025",
    }
    process_structured_data(sample_data)

# File: mfc/mfc/simulation.py

import os
import json
import time
import random
import logging
import numpy as np
from typing import Dict, Any, Tuple, List

from mfc.mfc.encoders.node_encoder import NodeEncoder
from mfc.mfc.modules.feedback_aggregator import FeedbackAggregator
from mfc.mfc.modules.deephydra_anomaly_detector import DeepHYDRAAnomalyDetector
from mfc.mfc.modules.decision_making import DecisionMakingModule
from mfc.mfc.modules.communication import CommunicationModule
from mfc.mfc.mfc_manager import MFCManager
from mfc.agents.CA_agent import CAAgent
from mfc.mfc.data_generation import generate_synthetic_time_series_data
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_system(config_path: str = '../config.json') -> Tuple[MFCManager, List[Dict[str, Any]]]:
    """
    Initializes the MFC system components from a configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Tuple[MFCManager, List[Dict[str, Any]]]: Instances of MFCManager and list of tasks.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize Node Encoder
    node_encoder = NodeEncoder(model_name='all-MiniLM-L6-v2')

    # Initialize DeepHYDRA Anomaly Detector (using Z-score for POC)
    anomaly_detector = DeepHYDRAAnomalyDetector()

    # Initialize Feedback Aggregator with Anomaly Detector
    feedback_aggregator = FeedbackAggregator(anomaly_detector=anomaly_detector)

    # Initialize Decision-Making Module (simple prioritization based on urgency)
    decision_making = DecisionMakingModule(resource_inventory=config['resource_inventory'])

    # Initialize Communication Module (basic message passing)
    communication_module = CommunicationModule(protocol_settings=config['communication_protocol'])

    # Initialize MFC Manager
    mfc_manager = MFCManager(
        config=config,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        feedback_aggregator=feedback_aggregator,
        decision_making=decision_making,
        communication_module=communication_module,
        node_encoder=node_encoder
    )

    # Initialize CA Agents with diverse behaviors
    for ca_config in config['agents']:
        agent = CAAgent(
            unique_id=ca_config['unique_id'],
            model=mfc_manager,
            initial_state=ca_config.get('initial_state', 0),
            initial_resource=ca_config.get('initial_resource', 5),
            behavior=ca_config.get('behavior', 'default'),
            base_model=ca_config.get('base_model'),
            role=ca_config.get('role'),
            tools=ca_config.get('tools', []),
            expertise_level=ca_config.get('expertise_level', 1),
            current_workload=ca_config.get('current_workload', 1),
            reliability_score=ca_config.get('reliability_score', 1.0),
            latency=ca_config.get('latency', 0.1),
            error_rate=ca_config.get('error_rate', 0.0),
            cost_per_task=ca_config.get('cost_per_task', 0.0)
        )
        mfc_manager.add_agent(agent)

    # Generate Tasks
    tasks = generate_tasks(num_tasks=5)  # Reduced number for POC

    return mfc_manager, tasks

def generate_tasks(num_tasks: int = 5) -> List[Dict[str, Any]]:
    """
    Generates a list of tasks with predefined attributes for the POC.

    Args:
        num_tasks (int): The number of tasks to generate.

    Returns:
        List[Dict[str, Any]]: A list of task dictionaries.
    """
    tasks = []
    for i in range(num_tasks):
        task = {
            "id": f"Task{i+1}",
            "type": random.choice(["TypeA", "TypeB", "TypeC"]),
            "resource_requirements": {
                "CPU": random.randint(1, 2),  # Simplified for POC
                "Memory": random.randint(1, 8),
                "Storage": random.randint(10, 50)
            },
            "deadline": f"2025-02-{random.randint(10, 28)}",
            "dependencies": [],  # Simplified for POC
            "priority": random.choice(["High", "Medium", "Low"]),
            "computational_complexity": random.randint(1, 3),
            "memory_footprint": random.randint(1, 8),
            "data_locality": random.choice(["Local", "Edge", "Cloud"]),
            "security_level": random.choice(["Confidential", "Restricted", "Public"]),
            "urgency_score": round(random.uniform(0.7, 1.0), 2),
            "expected_value": round(random.uniform(0.5, 1.0), 2),
            "precedence_relations": []
        }
        tasks.append(task)
    return tasks

def plot_simulation_metrics(resource_utilization, anomaly_counts):
    """
    Plots resource utilization and anomaly counts over simulation steps.

    Args:
        resource_utilization (List[int]): Total CPU utilization per step.
        anomaly_counts (List[int]): Number of anomalies detected per step.
    """
    steps = range(1, len(resource_utilization) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Resource Utilization
    plt.subplot(1, 2, 1)
    plt.plot(steps, resource_utilization, marker='o')
    plt.title('Total CPU Utilization Over Steps')
    plt.xlabel('Simulation Step')
    plt.ylabel('CPU Units')

    # Plot Anomaly Counts
    plt.subplot(1, 2, 2)
    plt.plot(steps, anomaly_counts, marker='x', color='red')
    plt.title('Anomalies Detected Over Steps')
    plt.xlabel('Simulation Step')
    plt.ylabel('Number of Anomalies')

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('simulation_metrics_step.png')
    logging.info("Simulation metrics saved to 'simulation_metrics_step.png'")

    # Optionally, clear the figure to free memory
    plt.close()

# Modify run_simulation to collect metrics
def run_simulation_with_metrics(mfc_manager: MFCManager, tasks: List[Dict[str, Any]], steps: int = 20):
    resource_utilization = []
    anomaly_counts = []

    for step in range(steps):
        logging.info(f"--- Starting MFC Step {step + 1} ---")
        try:
            # Encode agents and tasks
            agent_embeddings = mfc_manager.encode_agents()
            task_embeddings = mfc_manager.encode_tasks(tasks)

            # Distribute rules
            mfc_manager.distribute_rules()

            # Collect and aggregate feedback
            feedback = mfc_manager.collect_feedback()

            # Analyze patterns
            mfc_manager.analyze_patterns(feedback)

            # Adjust resource allocation
            mfc_manager.adjust_resource_allocation()

            # Prioritize tasks
            prioritized_tasks = mfc_manager.decision_making.prioritize_tasks(task_embeddings, agent_embeddings)

            # Allocate resources based on prioritized tasks
            allocation = mfc_manager.decision_making.allocate_resources(prioritized_tasks, agent_embeddings)

            # Communicate allocations to agents
            for task_id, resources in allocation.items():
                message = {
                    "task_id": task_id,
                    "allocated_resources": resources
                }
                if mfc_manager.agents:
                    recipient_id = random.choice(list(mfc_manager.agents.keys()))
                    mfc_manager.communication_module.send_message(recipient_id, message)
                    logging.info(f"Allocated {resources} to {recipient_id} for {task_id}")

            # Update agents
            for agent in mfc_manager.agents.values():
                neighbors = agent.get_neighbors()
                agent.step(neighbors)

            # Log aggregated feedback and anomalies
            logging.info(f"Aggregated Feedback: {feedback}")

            # Collect metrics
            total_cpu = sum([agent.resource for agent in mfc_manager.agents.values()])
            resource_utilization.append(total_cpu)

            anomalies = feedback.get("anomalies", [])
            anomaly_counts.append(len(anomalies))

        except Exception as e:
            logging.error(f"Error during simulation step {step + 1}: {e}")

        logging.info(f"--- Completed MFC Step {step + 1} ---\n")

    # Plot the collected metrics
    plot_simulation_metrics(resource_utilization, anomaly_counts)

if __name__ == "__main__":
    # Initialize system
    mfc_manager, tasks = initialize_system()

    # Run simulation with metrics
    run_simulation_with_metrics(mfc_manager, tasks, steps=20)
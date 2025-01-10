# Master Feedback Controller (MFC) Project

## **Project Overview**

The Master Feedback Controller (MFC) is designed to manage and coordinate multiple Cellular Automaton (CA) agents, ensuring efficient task allocation, resource management, and system robustness. This project leverages advanced encoding techniques and anomaly detection mechanisms to maintain optimal performance.

## **File Structure**

mfc/
│
├── mfc/
│   ├── __init__.py
│   ├── encoders/
│   │   └── node_encoder.py
│   ├── modules/
│   │   ├── feedback_aggregator.py
│   │   ├── deephydra_anomaly_detector.py
│   │   ├── decision_making.py
│   │   └── communication.py
│   ├── CA_agent.py
│   ├── mfc_manager.py
│   ├── simulation.py
│   ├── time_window_manager.py
│   └── agent_task_features_spec.md
│
├── README.md
└── requirements.txt

## **Setup Instructions**

1. **Clone the Repository:**

   git clone https://github.com/yourusername/mfc_project.git
   cd mfc_project/mfc

2. **Install Dependencies:**

   Ensure you have Python 3.7 or later installed. Install the required packages using pip:

   pip install -r requirements.txt

   **`requirements.txt`:**

   sentence-transformers
   numpy
   pandas
   logging

3. **Set Environment Variables:**

   The simulation requires an OpenAI API key for language model interactions. Set the `OPENAI_API_KEY` environment variable:

   export OPENAI_API_KEY='your-api-key-here'

4. **Run the Simulation:**

   Execute the simulation script to start the MFC:

   python simulation.py

## **Module Descriptions**

- **encoders/node_encoder.py:**
  Contains the `NodeEncoder` class responsible for encoding agent and task features into numerical embeddings using SentenceTransformer models.

- **modules/feedback_aggregator.py:**
  Implements the `FeedbackAggregator` class that aggregates feedback from multiple CAs and integrates anomaly detection.

- **modules/deephydra_anomaly_detector.py:**
  Defines the `DeepHYDRAAnomalyDetector` class that integrates DeepHYDRA for detecting anomalies in aggregated feedback.

- **modules/decision_making.py:**
  Implements the `DecisionMakingModule` class, which includes task prioritization using the IPRO algorithm and resource allocation based on prioritized tasks.

- **modules/communication.py:**
  Defines the `CommunicationModule` class that handles communication between the MFC and CAs using the AC2C protocol.

- **CA_agent.py:**
  Defines the `CAAgent` class, representing individual Cellular Automaton agents with specific behaviors and states.

- **mfc_manager.py:**
  Manages the overall coordination of agents, encoding processes, resource allocation, and communication within the MFC.

- **simulation.py:**
  The main script to initialize components, create agents, define tasks, and run the simulation for a specified number of steps.

- **time_window_manager.py:**
  Manages time-based windows for aggregating and analyzing feedback data.

- **agent_task_features_spec.md:**
  Detailed specification of agent and task features, including their descriptions and representation formats.

## **Next Steps**

1. **Finalize Agent and Task Features:**
   - Collaborate to ensure all necessary features are included and accurately represented.
   - Update `agent_task_features_spec.md` as needed based on further discussions.

2. **Implement Core MFC Modules:**
   - Complete the implementation of the `FeedbackAggregator`, `DecisionMakingModule`, and `CommunicationModule` classes.

3. **Develop Simple CAs:**
   - Create basic CA agents with simple functionalities to test the core mechanisms of the MFC.

4. **Enhance Simulation Environment:**
   - Add more complex tasks, agents, and environment factors to the simulation.
   - Implement realistic failure scenarios and anomaly injections.

5. **Integrate Security Measures:**
   - Ensure all communication is secured.
   - Regularly update the `security_audit_report.md` as new security features are implemented.

6. **Continuous Integration:**
   - Set up CI pipelines using tools like GitHub Actions to automate testing and deployment.

## **Contact**

For any questions or assistance, please reach out to Mike or Gemini via direct chat.

# Master Feedback Controller (MFC) Project

## Node Encoder Module

The `NodeEncoder` class is responsible for encoding agent and task features into numerical embeddings suitable for downstream processing within the MFC. This module utilizes pretrained SentenceTransformer models for generating text-based embeddings and combines them with numerical features.

### Setup Instructions

1. **Install Dependencies:**

    Ensure you have the necessary Python packages installed. You can install them using `pip`:

    ```bash
    pip install sentence-transformers numpy pandas logging
    ```

2. **Initialize the NodeEncoder:**

    ```python
    from mfc.mfc.encoders.node_encoder import NodeEncoder

    encoder = NodeEncoder(model_name='all-MiniLM-L6-v2')
    ```

3. **Encoding Agent Features:**

    ```python
    agent_data = {
        "Base Model": "GPT-4",
        "Role/Expertise": "Programmer",
        "Current State": {
            "Task": 3,
            "Resources": 5
        },
        "Available Plugins/Tools": ["Python Compiler", "File Reader"],
        "Expertise Level": "Intermediate",
        "Current Workload": 4,
        "Reliability Score": 0.95,
        "Latency": 0.2,
        "Error Rate": 0.01,
        "Cost Per Task": 0.5
    }

    agent_embedding = encoder.encode_agent(agent_data)
    ```

4. **Encoding Task Features:**

    ```python
    task_data = {
        "Type": "Development",
        "Resource Requirements": {
            "CPU": 4,
            "Memory": 16,
            "Storage": 100
        },
        "Deadline": "2025-02-01",
        "Dependencies": ["Task 2"],
        "Priority": "High",
        "Computational Complexity": 3,
        "Memory Footprint": 8,
        "Data Locality": "Local",
        "Security Level": "Confidential",
        "Urgency Score": 0.9,
        "Expected Value": 0.8,
        "Precedence Relations": ["Task1 -> Task2"]
    }

    task_embedding = encoder.encode_task(task_data)
    ```

5. **Running Unit Tests:**

    To ensure the `NodeEncoder` is functioning correctly, run the provided unit tests:

    ```bash
    python test_node_encoder.py
    ```

### **Notes**

*   The current implementation uses the `all-MiniLM-L6-v2` model for generating sentence embeddings. This can be changed by specifying a different `model_name` when initializing the `NodeEncoder`.
*   The `encode_graph` method is a placeholder for future implementations involving graph-level feature encoding.
*   Ensure that the numerical features provided in `agent_data` and `task_data` are appropriately scaled and normalized as needed for your specific use case.
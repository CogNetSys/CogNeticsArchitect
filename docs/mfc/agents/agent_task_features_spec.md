# Agent and Task Features Specification

## **Agent Features**

1. **Base Model**
    *   **Description:** The foundational language model used by the agent.
    *   **Data Type:** Categorical
    *   **Values:** `GPT-4`, `LLaMA-2`, `LLaMA-3`, `Mistral-7B`, `Mixtral-8x7B`, `Gemini-1.5 Pro` etc.
    *   **Encoding:** One-hot encoding
    *   **Rationale:** Captures the underlying capabilities and limitations of the agent's language processing.

2. **Role/Expertise**
    *   **Description:** The specialized function or expertise area of the agent.
    *   **Data Type:** Categorical
    *   **Values:** `Programmer`, `Mathematician`, `Data Analyst`, `Domain Expert`, `QA Tester`, `Manager`, etc. (extensible based on use-case)
    *   **Encoding:** One-hot encoding
    *   **Rationale:**  Indicates the agent's intended function within the multi-agent system.

3. **Current State**
    *   **Description:** The agent's current operational status and relevant context.
    *   **Data Type:** Mixed (numerical, categorical, text)
    *   **Components:**
        *   **Task ID:** (Numerical) Unique identifier of the task currently being executed, if any.
        *   **Task Status:** (Categorical) `Idle`, `Working`, `Waiting`, `Error`, `Completed`
        *   **Resources:** (Numerical) Current resource utilization (e.g., CPU %, memory %, bandwidth usage).
        *   **Assigned CA:** (Categorical) The Cellular Automata (CA) the agent is currently assigned to.
        *   **Location:** (Numerical, if applicable) Physical or virtual location of the agent.
        *   **Timestamp:** (Numerical) Time of the last state update.
    *   **Encoding:** Numerical values directly represented, categorical values one-hot encoded, free text (if any) embedded using a sentence transformer.
    *   **Rationale:** Provides a snapshot of the agent's current operational context, workload, and resource usage.

4. **Available Plugins/Tools**
    *   **Description:** Extensions or tools that the agent can utilize to perform tasks.
    *   **Data Type:** List of categorical variables
    *   **Examples:** `Python Compiler`, `Database Connector`, `Web Searcher`, `Image Analyzer`, `API Connector`
    *   **Encoding:** Multi-hot encoding (binary vector indicating presence/absence of each tool).
    *   **Rationale:** Indicates the agent's capabilities beyond its core language model.

5. **Expertise Level**
    *   **Description:** The level of expertise or proficiency of the agent in its assigned role.
    *   **Data Type:** Categorical
    *   **Values:** `Novice`, `Intermediate`, `Advanced`, `Expert`
    *   **Encoding:** Ordinal encoding (numerical representation reflecting the hierarchy of expertise).
    *   **Rationale:** Influences task allocation, collaboration strategies, and confidence weighting of agent outputs.

6. **Current Workload**
    *   **Description:** A measure of how busy the agent is, reflecting its current task queue and processing load.
    *   **Data Type:** Numerical
    *   **Values:** 0-100 (representing percentage of maximum workload capacity).
    *   **Encoding:** Direct numerical representation, potentially normalized.
    *   **Rationale:** Helps in dynamic task allocation and load balancing across agents.

7. **Reliability Score**
    *   **Description:** A measure of the agent's past performance, accuracy, and dependability.
    *   **Data Type:** Numerical
    *   **Values:** 0.0 - 1.0 (representing a probability or confidence score).
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:** Used to assess agent trustworthiness and weight their contributions in collaborative tasks. It can be updated based on feedback and performance evaluations.

8. **Latency**
    *   **Description:** Average response time of the agent.
    *   **Data Type:** Numerical
    *   **Values:** Time in milliseconds (ms).
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:**  Crucial for real-time applications; influences task allocation and communication strategies.

9. **Error Rate**
    *   **Description:** Frequency of errors or failures in task execution.
    *   **Data Type:** Numerical
    *   **Values:** Percentage of errors over a given period or number of tasks.
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:** Used to assess agent reliability and identify potential issues requiring intervention.

10. **Cost Per Task**
    *   **Description:** Cost associated with utilizing the agent, which could include operational costs, API usage fees, or other expenses.
    *   **Data Type:** Numerical
    *   **Values:** Cost in appropriate units (e.g., USD, credits).
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:**  Important for budget management and resource optimization within the MFC.

11. **Resources**
    * **Description:** Current resource utilization (e.g., CPU, memory, storage)
    * **Data Type:** Numerical
    * **Values:**: Units of resource.
    * **Encoding**: Direct numerical representation, normalized
    * **Rationale:**: Important to manage resources.

## **Task Features**

1. **Type**
    *   **Description:** The category or nature of the task.
    *   **Data Type:** Categorical
    *   **Values:** `Code Generation`, `Code Debugging`, `Code Optimization`, `Data Analysis`, `Text Summarization`, `Question Answering`, `Document Drafting`, `Image Generation`, `Logical Reasoning`, `Mathematical Computation`, `Information Retrieval` etc.
    *   **Encoding:** One-hot encoding.
    *   **Rationale:** Determines task-specific processing and agent assignment.

2. **Resource Requirements**
    *   **Description:** Computational and resource needs to execute the task.
    *   **Data Type:** Mixed (numerical)
    *   **Components:**
        *   **CPU:** (Numerical) Estimated CPU cores or processing power needed.
        *   **Memory:** (Numerical) Estimated RAM required (e.g., in GB).
        *   **Storage:** (Numerical) Estimated storage space needed (e.g., in GB).
        *   **GPU:** (Numerical) Number of GPUs needed, if any.
        *   **Bandwidth:** (Numerical) Estimated network bandwidth required (e.g., in Mbps).
    *   **Encoding:** Direct numerical representation for each resource type.
    *   **Rationale:** Informs resource allocation and task scheduling decisions.

3. **Deadline**
    *   **Description:** The time frame within which the task should be completed.
    *   **Data Type:** Numerical or Timestamp
    *   **Values:** Time in seconds, minutes, hours, or a specific timestamp.
    *   **Encoding:** Direct numerical representation (for time duration) or timestamp encoding.
    *   **Rationale:** Critical for real-time task management and prioritization.

4. **Dependencies**
    *   **Description:** Other tasks or components that the current task relies on.
    *   **Data Type:** List of task IDs or component identifiers.
    *   **Values:** `Task 1`, `Task 3`, `Component A`, etc.
    *   **Encoding:** Multi-hot encoding or sequence of task/component embeddings.
    *   **Rationale:**  Ensures tasks are executed in the correct order and that dependencies are satisfied before a task begins.

5. **Priority**
    *   **Description:** The importance level of the task relative to others.
    *   **Data Type:** Categorical
    *   **Values:** `High`, `Medium`, `Low` (or a numerical scale).
    *   **Encoding:** Ordinal encoding (numerical representation reflecting the priority level).
    *   **Rationale:**  Guides task prioritization and resource allocation, especially in resource-constrained scenarios.

6. **Computational Complexity**
    *   **Description:** An estimate of the computational resources required for the task.
    *   **Data Type:** Numerical or Categorical
    *   **Values:** Big O notation (e.g., `O(n)`, `O(n log n)`) or complexity levels (e.g., `Low`, `Medium`, `High`).
    *   **Encoding:** Direct numerical representation (if using Big O estimates) or ordinal encoding.
    *   **Rationale:** Helps in selecting appropriate agents and allocating sufficient resources.

7. **Memory Footprint**
    *   **Description:** An estimate of the memory needed during task execution.
    *   **Data Type:** Numerical
    *   **Values:** Memory size in GB.
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:**  Informs memory allocation decisions, especially for tasks with large memory requirements.

8. **Data Locality**
    *   **Description:** Information about where the required data for the task is located.
    *   **Data Type:** Categorical or Numerical
    *   **Values:** `Local`, `Edge`, `Cloud`, specific location identifiers, or distance metrics.
    *   **Encoding:** One-hot encoding for location categories or direct numerical representation for distances.
    *   **Rationale:**  Guides task placement decisions to minimize data transfer costs and latency.

9. **Security Level**
    *   **Description:** Sensitivity level of the task or data involved.
    *   **Data Type:** Categorical
    *   **Values:** `Confidential`, `Restricted`, `Public`, or a numerical scale.
    *   **Encoding:** Ordinal encoding.
    *   **Rationale:**  Ensures that tasks with high security requirements are handled by authorized agents and processed in secure environments.

10. **Urgency Score**
    *   **Description:** A numerical representation of how time-sensitive the task is.
    *   **Data Type:** Numerical
    *   **Values:** 0.0 - 1.0 (higher values indicate greater urgency).
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:**  Helps prioritize tasks based on their time criticality, especially in dynamic environments.

11. **Expected Value**
    *   **Description:** A measure of how valuable the task is to the overall goal.
    *   **Data Type:** Numerical
    *   **Values:** 0.0 - 1.0 (higher values indicate greater importance).
    *   **Encoding:** Direct numerical representation.
    *   **Rationale:**  Used to make trade-off decisions when allocating resources and prioritizing tasks.

12. **Precedence Relations**
    *   **Description:** An encoding of the dependencies between tasks.
    *   **Data Type:** List of task IDs or a dependency graph.
    *   **Values:** `Task1 -> Task2` (Task2 depends on Task1).
    *   **Encoding:** Adjacency matrix or list of directed edges representing dependencies.
    *   **Rationale:** Ensures that tasks are executed in the correct order and that dependencies are satisfied.

## **Feature Representation**

- **Embeddings:** Numerical representations capturing the semantic meaning of textual features.
- **One-Hot Encodings:** Binary vectors representing categorical features.
- **Numerical Values:** Direct numerical representations for resource levels and priorities.

## **Encoding Schemes**

- **Textual Features**: Encoded using the all-MiniLM-L6-v2 Sentence Transformer model, which produces 384-dimensional embeddings.

- **Numerical Features**: Directly represented as numerical values, normalized to a common scale if needed.

- **Categorical Features**:
    - One-hot encoding for nominal features (no inherent order).
    - Ordinal encoding for ordinal features (ordered categories).
- **List/Sequence Features**: Multi-hot encoding for lists of categorical features (e.g., available tools) or sequences of embeddings for task dependencies.
- **Timestamp**: Converted to a numerical representation, such as seconds since the epoch.

### Final Encoding Scheme Specification for Each Feature

We will be using this specific encoding scheme for each feature in the NodeEncoder:
```python
# Example Agent Data for Encoding
agent_example = {
    "Base Model": "GPT-4",  # One-hot encode
    "Role/Expertise": "Programmer",  # One-hot encode
    "Current State": {
        "Task": 3,  # Direct numerical representation
        "Resources": 5  # Direct numerical representation
    },
    "Available Plugins/Tools": ["Python Compiler", "File Reader"],  # Multi-hot encode
    "Expertise Level": 2,  # Ordinal encode
    "Current Workload": 4,  # Direct numerical representation
    "Reliability Score": 0.95,  # Direct numerical representation
    "Latency": 0.2,  # Direct numerical representation
    "Error Rate": 0.01,  # Direct numerical representation
    "Cost Per Task": 0.5  # Direct numerical representation
}

# Example Task Data for Encoding
task_example = {
    "Type": "Development",  # One-hot encode
    "Resource Requirements": {
        "CPU": 4,  # Direct numerical representation
        "Memory": 16,  # Direct numerical representation
        "Storage": 100  # Direct numerical representation
    },
    "Deadline": "2025-02-01",  # Timestamp to numerical
    "Dependencies": ["Task 2"],  # Sequence of task embeddings
    "Priority": "High",  # Ordinal encode
    "Computational Complexity": 3,  # Direct numerical representation or ordinal
    "Memory Footprint": 8,  # Direct numerical representation
    "Data Locality": "Local",  # One-hot encode
    "Security Level": "Confidential",  # Ordinal encode
    "Urgency Score": 0.9,  # Direct numerical representation
    "Expected Value": 0.8,  # Direct numerical representation
    "Precedence Relations": ["Task1 -> Task2"]  # Adjacency list representation
}
```
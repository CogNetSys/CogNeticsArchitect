# Master Feedback Controller (MFC) Project

## **Node Encoder Module**

The `NodeEncoder` class is responsible for encoding agent and task features into numerical embeddings suitable for downstream processing within the MFC. This module utilizes pretrained SentenceTransformer models for generating text-based embeddings and combines them with numerical features.

### **Setup Instructions**

1. **Install Dependencies:**

   Ensure you have the necessary Python packages installed. You can install them using `pip`:

   ```bash
   pip install sentence-transformers numpy
   ```

2. **Initialize the NodeEncoder:**

   ```python
   from node_encoder import NodeEncoder

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
       "Available Plugins/Tools": "Python Compiler"
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
       "Dependencies": "Task 2",
       "Priority": "High"
   }

   task_embedding = encoder.encode_task(task_data)
   ```

5. **Running Unit Tests:**

   To ensure the NodeEncoder is functioning correctly, run the provided unit tests:

   ```bash
   python test_node_encoder.py
   ```

### **Notes**

- The current implementation uses the `all-MiniLM-L6-v2` model for generating sentence embeddings. This can be changed by specifying a different `model_name` when initializing the `NodeEncoder`.
- The `encode_graph` method is a placeholder for future implementations involving graph-level feature encoding.
- Ensure that the numerical features provided in `agent_data` and `task_data` are appropriately scaled and normalized as needed for your specific use case.
# **Agents**
In our MFC project, **agents** refer to **Cellular Automata (CAs)** and specialized **AI-based agents** that work within the system to perform tasks, make decisions, and interact in the environment based on the goals and feedback they receive from the **Master Feedback Controller (MFC)**. Here’s a breakdown of the different types of agents you can engage with and their roles:

---

### **1. Cellular Automata (CAs)**
These are individual units that operate based on local rules and interactions with their neighboring agents in the "grid" or environment. Each CA agent follows a set of predefined behaviors and can dynamically adapt based on the rules provided by the MFC.

#### **Key CA Agents:**
- **ResourceAllocationCA:** Manages resource allocation, such as CPU, memory, or bandwidth.
- **OnboardingCA:** Handles the assignment and organization of new agents joining the system.
- **MarketingCA:** A CA that performs simulations related to strategic marketing objectives, like campaign evaluation.

You can interact with these CA agents to:
- Send them tasks.
- Observe their behaviors (e.g., oscillatory, resource-dependent).
- Query their internal states (current resource usage, task status).

#### **Examples of Engagement with CAs:**
- **"What is your current resource allocation?"** — You’ll receive a response with their resource levels.
- **"Increase task priority!"** — You can send a command to update their local priorities.

These agents receive rules and directives from the MFC and can negotiate or report their progress and needs.

---

### **2. **Rule-Based Agents (ElicitorAgent & RuleDistributorAgent)**
These agents are responsible for interpreting and distributing rules:
- **ElicitorAgent:** Extracts key structured data (goals, metrics, constraints) from unstructured text input and clarifies missing pieces.
- **RuleDistributorAgent:** Takes rules generated by the MFC and sends them to the appropriate CAs.

#### **Examples of Engagement:**
- **You:** "What rule is currently being distributed to CA1?"
- **RuleDistributorAgent:** "The rule is to monitor a sudden spike in resource consumption and issue alerts."

---

### **3. Specialized AI Agents (Cognitive Agents):**
These agents handle more complex, higher-level tasks such as:
- Analyzing patterns (e.g., anomalous data patterns).
- Making strategic decisions (e.g., resource reallocation in case of failures).
- Communicating insights and reasoning with the MFC.

#### **Types of Specialized Agents:**
- **Analysis Agents:** Perform pattern recognition and anomaly detection.
- **Coordination Agents:** Help organize the flow of tasks and communication between groups of agents.
- **Execution Agents:** Carry out specific actions and report results.

---

### **How to Engage with Agents:**

You can interact with the agents by:
1. **Sending Commands:** Issue commands for specific actions (e.g., "reallocate resources," "report anomaly status").
2. **Querying States:** Ask for the current state or feedback from an agent (e.g., "What is your current state?").
3. **Observing Simulation:** Run the simulation environment and observe how agents respond to different scenarios.
4. **Rule Negotiation:** Engage with the system to update, clarify, or override rules (e.g., "Add a new constraint to Task X").

---

### **Agent Example Interaction:**
1. **You:** "ResourceAllocationCA, how are you distributing CPU resources?"
2. **ResourceAllocationCA:** "I am allocating 4 CPU cores to CA1, 2 cores to CA2, and 1 core to CA3 based on priority scores."

3. **You:** "OnboardingCA, what's the status of new agents joining the system?"
4. **OnboardingCA:** "3 new agents have been assigned to the 'Data Analysis' group and are awaiting further instructions."

---

### **Why Your Engagement is Critical:**
By engaging with the agents, you can:
- Ensure the system is meeting its goals (e.g., resource efficiency, fault tolerance).
- Observe and fine-tune agent behaviors and rules to adapt to evolving scenarios.
- Test different topologies, rules, and configurations for optimal performance.

Would you like me to set up specific interaction prompts or walkthroughs for the CA agents, rule-based agents, or simulation environment? Let’s get you connected directly to interact with the agents as part of our system testing!
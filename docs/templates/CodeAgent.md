# CodeAgent: Your Intelligent Coding Assistant

The `CodeAgent` is a powerful component of the `smolagents` library that brings the capabilities of large language models (LLMs) to the realm of code generation and execution. It's designed to be your intelligent coding assistant, capable of understanding natural language instructions, writing Python code to accomplish tasks, and even executing that code either locally or in a secure sandboxed environment.

## What Can CodeAgent Do?

At its core, a `CodeAgent` can:

1. **Generate Code:** Based on your instructions (the "task"), the `CodeAgent` uses a connected LLM to generate Python code snippets that attempt to solve the task.
2. **Execute Code:** It can execute the generated code, either locally using a restricted Python interpreter or remotely using the E2B sandbox for enhanced security.
3. **Leverage Tools:** You can extend the `CodeAgent`'s capabilities by providing it with custom "tools." These tools are essentially Python functions wrapped in a special `Tool` class, allowing the agent to call them as part of its problem-solving process.
4. **Manage State:** The agent can maintain a state, allowing you to pass in data or variables that the generated code can access. This is useful for providing context or working with user-provided files.
5. **Reason and Iterate:** The `CodeAgent` follows a "ReAct" (Reasoning and Acting) framework. It can analyze the results of its actions (code execution or tool use), reason about the next steps, and iterate until it arrives at a solution or reaches a maximum number of attempts.

## Getting Started

### Installation

Before using `CodeAgent`, make sure you have the `smolagents` library and its dependencies installed. If you intend to use Groq models, you'll also need the `litellm` package. For local code execution, the necessary packages are already included in smolagents. For using the `e2b` executor, you will need to install its package. You can install everything with pip:

```bash
pip install smolagents "litellm[groq]" e2b-code-interpreter python-dotenv pandas
```
### Setting Up Your Environment

Virtual Environment (Recommended): Create and activate a new virtual environment to isolate your project's dependencies:
```
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
```

Environment Variables: If you're using API-based models (like Groq, OpenAI, etc.), store your API keys in a .env file in your project's root directory:
```
GROQ_API_KEY=your_groq_api_key
E2B_API_KEY=your_e2b_api_key  # If using E2B
```

The python-dotenv package, which you installed earlier, will automatically load these variables into your environment when you run your script.

### Basic CodeAgent Usage

Here's a simple example of how to create and use a CodeAgent:
```
from smolagents import CodeAgent, LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the LLM
# Using the LiteLLMModel to access the Groq model
llm_model = LiteLLMModel(model_id="groq/llama3-70b-8192")

# 2. Initialize the CodeAgent
# No tools for now, just basic code generation and execution
agent = CodeAgent(model=llm_model, tools=[])

# 3. Run the agent with a simple task
task = "What is 2 multiplied by 234591?"
result = agent.run(task)
print(result)
```

#### Explanation:

Import necessary classes: CodeAgent and LiteLLMModel.

Load environment variables: load_dotenv() loads your API keys from the .env file.

Initialize the LLM: We create a LiteLLMModel instance, specifying the Groq model you want to use.

Initialize the CodeAgent: We create a CodeAgent, passing in the llm_model. For now, we don't give it any tools (tools=[]).

Run the Agent: The agent.run() method starts the agent's reasoning and action process. The agent will generate Python code to calculate 2 * 234591, execute it, and return the result.

### What Happens Under the Hood

When you run the agent:

Prompting: The CodeAgent combines its system prompt with the user-provided task to create a prompt for the LLM.

Code Generation: The LLM generates Python code that it believes will solve the task.

Code Execution: The generated code is executed (either locally or in the E2B sandbox, depending on your configuration).

Result: The output of the code execution is captured. If the code includes a final_answer tool, the agent considers the task complete. Otherwise, it might iterate further, using the output as feedback to refine its approach.

## Adding Tools to Your CodeAgent

Tools are what make your CodeAgent truly powerful. They extend the agent's capabilities beyond just generating code, allowing it to interact with external systems, APIs, databases, and more.

#### Creating a Custom Tool

Let's create a tool that can analyze data in a CSV file using the pandas library:
```
import pandas as pd
from smolagents import Tool, tool
from smolagents.default_tools import PythonInterpreterTool
```
#### Using the @tool decorator (easy way)
```
@tool
def analyze_csv_column(filepath: str, column: str) -> str:
    """
    Loads a CSV file, and analyzes a specific column.

    Args:
        filepath: The path to the CSV file.
        column: The name of the column to analyze.

    Returns:
        A string containing the analysis results (mean, median, standard deviation).
    """
    try:
        df = pd.read_csv(filepath)
        if column not in df.columns:
            return f"Error: Column '{column}' not found in the DataFrame."

        mean = df[column].mean()
        median = df[column].median()
        std_dev = df[column].std()

        result = (
            f"Data Analysis for column '{column}' in file '{filepath}':\n"
            f"  Mean: {mean:.2f}\n"
            f"  Median: {median:.2f}\n"
            f"  Standard Deviation: {std_dev:.2f}"
        )
        return result
    except FileNotFoundError:
        return f"Error: File not found at path: {filepath}"
    except Exception as e:
        return f"Error during data analysis: {e}"
```
#### OR, creating a tool by subclassing Tool (more control)
```
class DataAnalysisTool(Tool):
    name = "data_analyzer"
    description = "Loads a CSV file into a pandas DataFrame and provides basic analysis (e.g., mean, median, standard deviation) of a specified column."
    inputs = {
        "filepath": {
            "type": "string",
            "description": "Path to the CSV file.",
        },
        "column": {
            "type": "string",
            "description": "The name of the column to analyze.",
        },
    }
    output_type = "string"

    def forward(self, filepath: str, column: str) -> str:
        # ... (same implementation as in the @tool example) ...
```

#### Explanation:

- **@tool Decorator:** The @tool decorator is a convenient way to define simple tools.

- **Tool Class:** For more complex tools or if you prefer a class-based approach, you can subclass Tool.

- **name:** The name of your tool (used by the agent to refer to it).

- **description:** A clear description of what your tool does. This is essential for the LLM to understand how to use it.

- **inputs:** A dictionary defining the input parameters (if any), their types, and descriptions.

- **output_type:** Specifies the type of the output returned by the tool.

- **forward():** This method contains the actual logic of your tool. It takes the input arguments, performs the necessary operations, and returns the result.

## Using Tools in Your CodeAgent

Now, let's see how to use this DataAnalysisTool (or the one created with @tool) in your CodeAgent:
```
... (import statements, DataAnalysisTool definition) ...

# Initialize the Groq model
your_groq_model = LiteLLMModel(model_id="groq/llama3-70b-8192")

# Create a tool instance
data_analysis_tool = DataAnalysisTool()

# Initialize the CodeAgent with the tool
code_agent = CodeAgent(
    model=your_groq_model,
    tools=[data_analysis_tool, PythonInterpreterTool()],  # Include the tool here
    additional_authorized_imports=["pandas"],  # Add pandas to authorized imports
    # use_e2b_executor=True  # Uncomment to use E2B
)

# Create a sample CSV file (or use your own)
csv_data = """product,price,quantity
apple,1.0,10
banana,0.5,20
cherry,0.2,50
"""
with open("sales_data.csv", "w") as f:
    f.write(csv_data)

### Run the agent with a task that uses the tool
task = (
    "Analyze the 'price' column in the dataset 'sales_data.csv' "
    "and tell me the mean, median, and standard deviation."
)
result = code_agent.run(task, additional_args={"sales_data.csv": "sales_data.csv"})
print(result)
```

#### Explanation:

Tool Instance: You create an instance of your DataAnalysisTool.

- **tools List:** You pass the data_analysis_tool instance in the tools list when initializing the CodeAgent. This makes the tool available to the agent.

- **additional_authorized_imports:** Since the DataAnalysisTool uses pandas, we add it to the list of authorized imports.

- **Task:** The task now instructs the agent to use the tool to analyze the CSV data.

### How the Agent Uses the Tool

- **Prompting:** The CodeAgent constructs a prompt for the LLM that includes:

The system prompt (which describes the agent's capabilities and how to use tools).

### The user's task.

Descriptions of the available tools (including your DataAnalysisTool).

- **Code Generation:** The LLM generates Python code that it believes will solve the task. This code will likely include a call to your data_analysis_tool.forward() method.

- **Code Execution:** The generated code is executed. When the data_analysis_tool.forward() call is encountered, your tool's forward method is executed.

- **Result:** The output of the tool (the analysis results) is captured and potentially used by the agent in further steps or included in the final answer.

## Advanced Usage

### Multi-Agent Systems with ManagedAgent

You can create more complex systems by having CodeAgents manage other agents. This is useful for breaking down large tasks into smaller, more specialized subtasks.

Here's an example of a CodeAgent managing a ToolCallingAgent that performs web searches:
```
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    Tool,
    tool,
)
from dotenv import load_dotenv
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException

load_dotenv()

# Define the VisitWebpageTool
@tool
def visit_webpage(url: str) -> str:
    # ... (Implementation of visit_webpage) ...

# Initialize the Groq model using LiteLLM
model = LiteLLMModel(model_id="groq/llama3-70b-8192")

# Create the web search agent
web_search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_iterations=5,
)

# Create the managed web search agent
managed_web_search_agent = ManagedAgent(
    agent=web_search_agent,
    name="web_searcher",
    description="Performs web searches and can visit web pages to extract information. Provide a detailed query.",
    managed_agent_prompt="""You're a helpful agent named '{name}'.
You have been submitted this task by your manager.
---
Task:
{task}
---
You're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible to give them a clear understanding of the answer.

Your final_answer WILL HAVE to contain these parts:
### 1. Task outcome (short version):
### 2. Task outcome (extremely detailed version):
### 3. Additional context (if relevant):

Make sure the content you provide in final_answer is a string.
Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.
And even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback.
{additional_prompting}""",
)

# Create the manager agent
manager_agent = CodeAgent(
    model=model,
    tools=[],  # Explicitly pass an empty list of tools
    managed_agents=[managed_web_search_agent],
    max_iterations=7,
    additional_authorized_imports=["time", "numpy", "pandas"],
    # use_e2b_executor=True # Removed E2B
)

# Run the manager agent
task = "What is the current population of France? Please use web search to find the latest information."
result = manager_agent.run(task)
print(result)
```

#### Explanation:

- **ToolCallingAgent:** We create a ToolCallingAgent specifically for web search tasks. It has access to the DuckDuckGoSearchTool and visit_webpage tools.

- **ManagedAgent:** We wrap the web_search_agent in a ManagedAgent. This gives it a name ("web_searcher") and a description that will be used by the manager_agent to understand its capabilities. We also provide a custom prompt to the managed agent, to specify how it should behave and format its output.

- **CodeAgent (Manager):** The manager_agent is a CodeAgent that doesn't have direct tools (tools=[]) but manages the managed_web_search_agent.

- **Task Delegation:** When you run the manager_agent, it will likely delegate the web search part of the task to the managed_web_search_agent.

### Customizing the System Prompt

You can customize the CodeAgent's behavior by providing your own system_prompt. This prompt is used to instruct the underlying LLM.

**Important:** If you override the system_prompt, make sure to include placeholders for:

- **{{tool_descriptions}}:** This is where the descriptions of the available tools will be inserted.

- **{{tool_names}}:** This is where the names of the available tools will be inserted.

- **{{managed_agents_descriptions}}:** This is where the descriptions of managed agents will be inserted (if any).

- **{{authorized_imports}}:** This is where the list of authorized imports will be inserted.

Here is an example of how to create your own system prompt. This prompt needs to be written in jinja2:
```
custom_system_prompt = """
You are a powerful coding assistant.

You should follow these guidelines:

-   Generate Python code that fulfills the user's request.
-   The code should be correct, efficient, and easy to understand.
-   You can use the following tools:

    {{tool_descriptions}}

-   The tools are just functions, DO NOT pass the arguments as a dictionary but directly as arguments.
-   If you need to use variables, make sure they are defined in your code.
-   You can use the following imports:

    {{authorized_imports}}

-   Do not import other modules.
-   If you need to use other modules, ask the user to provide them in a new tool.
-   Your final answer should be the result of the final computation, and you must provide it using the `final_answer` tool.

-   Make your code as simple as possible, avoid creating classes or functions if not needed.
-   If the task is complex, break it down into smaller, manageable steps using multiple code blocks.

{{managed_agents_descriptions}}

Now, let's start!
"""

code_agent = CodeAgent(
    model=your_groq_model,
    tools=[python_interpreter_tool, data_analysis_tool],
    system_prompt=custom_system_prompt
)
```
## Using the Gradio Interface

You can easily launch a Gradio interface to interact with your CodeAgent using the GradioUI class:
```
from smolagents import GradioUI

# ... (your agent initialization) ...

ui = GradioUI(code_agent)
ui.launch()
```

This will start a Gradio web server that allows you to chat with your agent, providing a more user-friendly way to interact with it.

## Tips for Building Effective Code Agents

- **Clear and Concise Tool Descriptions:** The LLM relies on the tool descriptions to understand how to use them. Write clear, concise, and accurate descriptions.

- **Modular Design:** Break down complex tasks into smaller, more manageable subtasks. This makes it easier for the agent to reason and generate correct code. Consider using managed agents for this.

- **Iterative Development:** Building a successful agent often requires experimentation. Start with a simple agent and gradually add complexity. Use the logs and the Gradio UI to debug and refine your agent's behavior.

- **Error Handling:** Implement robust error handling in your tools and agent logic to gracefully handle unexpected situations.

- **Security:** If using the local PythonInterpreter, be very cautious about the code that is executed. Avoid running untrusted code. Consider using the E2B sandbox for enhanced security.

- **Model Selection:** Choose an LLM that is well-suited for code generation and the complexity of your tasks. Larger models often perform better but might be slower.

## Conclusion

`CodeAgent` provides a powerful and flexible way to build intelligent coding assistants. By combining the reasoning capabilities of LLMs with the ability to generate and execute code, you can create agents that automate a wide range of programming tasks. Remember to experiment, iterate, and refine your agent's design to achieve the best results.

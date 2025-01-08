# 🌐 WebSearchAgent: Your Ultimate Multi-Agent Web Research Assistant

The `WebSearchAgent.py` script demonstrates a powerful multi-agent system built using the `smolagents` library. This system combines the reasoning capabilities of a **manager agent** (`CodeAgent`) with the specialized skills of a **web search agent** (`ToolCallingAgent`) to perform sophisticated web research and information retrieval tasks.

## 🌟 Architecture Overview

The system has a hierarchical structure, designed for efficiency and clarity:

1. **Manager Agent (`CodeAgent`)**: This is the top-level agent. It receives the user's task, orchestrates the overall process, and delegates subtasks to the managed agent. It's also capable of executing Python code if needed, but in this specific setup, its primary role is delegation and management.

2. **Managed Web Search Agent (`ManagedAgent`)**: This agent encapsulates the `web_search_agent` and provides a clear interface for the manager agent to interact with it. It acts as a specialized unit for web search operations. The `ManagedAgent` wrapper allows the manager to treat the web search functionality as a single, cohesive tool.

3. **Web Search Agent (`ToolCallingAgent`)**: This agent is responsible for the core web research functionality. It leverages the `DuckDuckGoSearchTool` to perform web searches and the `visit_webpage` tool to extract and process content from web pages. It communicates with the LLM using structured JSON-formatted tool calls, making its interactions precise and predictable.

**Visual Representation:**

    +-----------------+
    |  Manager Agent  |  (CodeAgent)
    +-----------------+
            ^
            | Delegates tasks to
            |
    +-----------------+
    | Managed Agent   |  (ManagedAgent)
    | (web_searcher)  |
    +-----------------+
            ^
            | Manages
            |
    +-----------------+
    | Web Search Agent|  (ToolCallingAgent)
    +-----------------+
        /           \
        /             \
    DuckDuckGoSearchTool       visit_webpage Tool

## 🛠️ Tools at Work

This system utilizes two main tools to accomplish web research tasks:

1. **`DuckDuckGoSearchTool`:** This tool performs a web search using DuckDuckGo. It takes a search query as input and returns a list of search results. Each result includes the title, URL, and a short snippet (body) of the web page.

2. **`visit_webpage`:** This tool fetches the content of a given URL, converts it into a clean Markdown format, and returns the processed text. It's crucial for extracting information from web pages found during the search.

## 🚀 Getting Started

### Prerequisites

Before running the `WebSearchAgent`, make sure you have the following libraries installed:

*   **`smolagents`:** The core library for building agents.
*   **`litellm` (if using Groq or other API-based models):** `pip install "litellm[groq]"`
*   **`requests`:** For making HTTP requests (used by `visit_webpage`).
*   **`markdownify`:** For converting HTML to Markdown.
*   **`duckduckgo-search`:** For performing DuckDuckGo searches.
*   **`python-dotenv`:** For managing environment variables.
*   **`huggingface_hub`**: For using an `HfApiModel`.

You can install them using this command:

    pip install smolagents "litellm[groq]" python-dotenv requests markdownify duckduckgo-search huggingface_hub

### Environment Setup

1. **Virtual Environment (Recommended):** It's strongly advised to create and activate a virtual environment to keep your project dependencies isolated:

    **Create a virtual environment:**

    python3 -m venv .venv

    **Activate the environment (on Linux/macOS):**

    source .venv/bin/activate

2. **API Keys:**
    *   If you're using Groq, set your `GROQ_API_KEY` in a `.env` file in your project's root directory:

        ```
        GROQ_API_KEY=your_groq_api_key
        ```

    *   If you're using the Hugging Face Inference API, you'll need a Hugging Face account and an API token. You can either set the `HF_TOKEN` environment variable or use `huggingface-cli login` to store your credentials.

## 👨‍💻 Code Walkthrough: `WebSearchAgent.py`

```python
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,  # or LiteLLMModel for Groq
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
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Initialize the model
# model = LiteLLMModel(model_id="groq/llama3-70b-8192") # For Groq
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")  # For HuggingFace Inference API

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
    tools=[],  # No need for extra tools here
    managed_agents=[managed_web_search_agent],
    max_iterations=7,
    additional_authorized_imports=["time", "numpy", "pandas"],
    use_e2b_executor=False,  # We are using local execution
)

# Run the manager agent
task = "What is the current population of America? Please use web search to find the latest information."
result = manager_agent.run(task)
print(result)
```

## Let's break down the key components:

### visit_webpage Tool
This tool is defined using the @tool decorator, making it easily discoverable by smolagents.

**`visit_webpage(url: str) -> str:`**
- Takes a URL as a string input (`url: str`).
- Returns a string (`-> str`) containing the Markdown content of the page or an error message.
- Uses the `requests` library to fetch the webpage content.
- Uses `markdownify` to convert the HTML to Markdown, providing a cleaner text representation for the LLM to process.
- Includes error handling with `try-except` blocks to catch issues like network problems or invalid URLs.


### Model Initialization
**`model = ...:`**
- You can choose between `LiteLLMModel` (for Groq or other providers via LiteLLM) or `HfApiModel` (for Hugging Face Inference API models).
- Make sure to set the correct `model_id`.


### Web Search Agent (`ToolCallingAgent`)
**`web_search_agent = ToolCallingAgent(...)`**
- **`tools=[DuckDuckGoSearchTool(), visit_webpage]`**: This agent is equipped with two tools: one for searching and one for fetching page content.
- **`model=model`**: The agent uses the specified LLM for reasoning and tool calling.
- **`max_iterations=5`**: Limits the agent to a maximum of 5 steps to prevent infinite loops.


### Managed Web Search Agent (`ManagedAgent`)
**`managed_web_search_agent = ManagedAgent(...)`**
- **`agent=web_search_agent`**: This wraps the `web_search_agent`, making it a manageable unit for the `manager_agent`.
- **`name="web_searcher"`**: Gives a descriptive name to the managed agent.
- **`description="..."`**: Provides a clear explanation of the managed agent's capabilities. This description is crucial for the `manager_agent` to understand when and how to delegate tasks.
- **`managed_agent_prompt`**: Contains the prompt that will be used with the managed agent. Special tags like `{name}`, `{task}`, and `{additional_prompting}` personalize its behavior.

### Manager Agent (`CodeAgent`)
**`manager_agent = CodeAgent(...)`**
- **`model=model`**: The manager agent also uses the same LLM.
- **`tools=[]`**: The manager agent doesn’t directly use any tools in this example. It relies on the `managed_web_search_agent`.
- **`managed_agents=[managed_web_search_agent]`**: This provides the list of managed agents. In this case, it's just the `web_searcher`.
- **`max_iterations=7`**: The manager agent can take up to 7 steps.
- **`additional_authorized_imports=[...]`**: Lists any extra Python modules the agent might need for code generation.
- **`use_e2b_executor=False`**: Set to `False` to use the local Python interpreter.

### Running the Agent
- **`task = ...`**: Defines the task for the agent.
- **`result = manager_agent.run(task)`**: Starts the agent execution.

## 🚀 Execution Flow
1. The `manager_agent` receives the task: _"What is the current population of America? Please use web search to find the latest information."_
2. The `manager_agent`'s LLM analyzes the task and delegates it to the `web_searcher` (the `managed_web_search_agent`).
3. The `managed_web_search_agent` receives the task and passes it to the `web_search_agent`.
4. The `web_search_agent` uses the `DuckDuckGoSearchTool` to perform a web search.
5. Based on the search results, the `web_search_agent` might decide to use the `visit_webpage` tool to get more information from specific URLs.
6. The `web_search_agent` iterates, making multiple tool calls until it gathers enough information.
7. The `web_search_agent` formulates a final answer using the `final_answer` tool, formatted based on the `managed_agent_prompt`.
8. The `managed_web_search_agent` returns the final answer to the `manager_agent`.
9. The `manager_agent` may perform further processing or simply return the final answer.

## 💡 Customization and Extensions
- **System Prompt**: Modify the default system prompt of the `manager_agent` or the `web_search_agent` to fine-tune their behavior. Include placeholders like `{{tool_descriptions}}`, `{{tool_names}}`, and `{{managed_agents_descriptions}}` if applicable.
- **Adding More Tools**: Extend the `web_search_agent`'s capabilities by adding tools (e.g., a calculator tool or data analysis tool).
- **More Complex Logic**: Create sophisticated agents to handle workflows, conditional logic, and error handling.
- **Different Models**: Experiment with different LLMs to find the best performer for your tasks.
- **E2B for Secure Execution**: For enhanced security, set `use_e2b_executor=True` in the `CodeAgent` and provide an E2B API key.

## 🎯 Conclusion
This guide demonstrates how to build a multi-agent system for web research using `smolagents`. By defining custom tools, creating specialized agents, and orchestrating them with a manager agent, you can create intelligent assistants capable of tackling complex, real-world tasks.

# 🌟 ToolCallingAgent Guide 🌟

The `ToolCallingAgent` is your gateway to building sophisticated, structured agents within the `smolagents` ecosystem. Unlike its sibling, the `CodeAgent`, which generates and executes code, the `ToolCallingAgent` focuses on orchestrating **JSON-formatted tool calls**. This approach offers greater control over agent behavior and seamless integration with systems that expect structured input.

This guide will equip you with the knowledge and practical examples to become a `ToolCallingAgent` pro.

## 💡 Why Choose ToolCallingAgent?

The `ToolCallingAgent` shines in scenarios where:

*   **Structured Interactions:** You need your agent to interact with tools or external systems using a well-defined, easily parsable format like JSON.
*   **Control and Validation:** You want more control over the agent's output and the ability to validate tool calls before execution.
*   **Integration:** You're integrating with APIs or services that expect structured data.
*   **Model Strengths:** You're using a model that excels at generating structured outputs like JSON but might not be as strong at general-purpose code generation.

## 🎯 Core Concepts

At its heart, the `ToolCallingAgent` operates on a few key principles:

1. **Reasoning and Planning:** The agent receives a task and uses its LLM brain to devise a plan, deciding which tools to use and in what order.
2. **Tool Call Generation:** Instead of writing code, the agent generates a JSON object representing a tool call. This JSON specifies the tool's name (`action`) and the arguments to pass to it (`action_input`).
3. **Tool Execution:** The `ToolCallingAgent` framework parses the JSON, identifies the corresponding tool, and executes it with the provided arguments.
4. **Observation and Iteration:** The agent receives the output (observation) from the tool execution. Based on this, it either provides a final answer (using the special `final_answer` tool) or continues to the next step, repeating the reasoning, tool call generation, and execution process.

## 🚀 Getting Started: Your First ToolCallingAgent

Let's build a simple agent that can greet users.

### Prerequisites

Make sure you have the necessary packages installed. You'll need `smolagents` and `litellm` if you are using API-based models like Groq:

**Install command:**

    pip install smolagents "litellm[groq]" python-dotenv

### Environment Setup

If you are using API-based models (like in this example with Groq), make sure to set your API keys as environment variables. You can use a .env file for this:

**In your .env file:**

    GROQ_API_KEY=your_groq_api_key

### Building the Agent

**Code Example:**

    from smolagents import ToolCallingAgent, LiteLLMModel, Tool, tool
    from dotenv import load_dotenv

    load_dotenv()

    # Define a sample tool using the @tool decorator
    @tool
    def greeting_tool(name: str) -> str:
        """Greets the person with the given name.

        Args:
            name: The name of the person to greet.
        """
        return f"Hello, {name}! It's nice to see you."

    # Initialize the Groq model
    your_groq_model = LiteLLMModel(model_id="groq/llama3-70b-8192")

    # Initialize the ToolCallingAgent
    tool_calling_agent = ToolCallingAgent(
        model=your_groq_model,
        tools=[greeting_tool],
        max_iterations=3
    )

    # Run the agent
    result = tool_calling_agent.run("Could you greet my friend Alice using the appropriate tool?")
    print(result)

**Explanation:**

1. **Import Necessary Classes:** We import `ToolCallingAgent`, `LiteLLMModel`, and `tool` from `smolagents`.
2. **Define a Tool:** We define a simple `greeting_tool` using the `@tool` decorator. This tool takes a name as input and returns a greeting string.
3. **Initialize the LLM:** We create a `LiteLLMModel` instance to use the Groq model.
4. **Initialize `ToolCallingAgent`:**
    *   `model`: We pass the `your_groq_model` instance.
    *   `tools`: We provide a list containing our `greeting_tool`.
    *   `max_iterations`: We set the maximum number of steps the agent can take to 3 (you can adjust this).
5. **Run the Agent:** The `run()` method starts the agent. The agent will:
    *   Receive the task: "Could you greet my friend Alice using the appropriate tool?"
    *   Generate a JSON-formatted tool call to use the `greeting_tool`.
    *   Execute the tool call.
    *   Generate a final response using the `final_answer` tool.

**Expected Output:**

The output will be the result of the `greeting_tool` being called with the name "Alice":

    Hello, Alice! It's nice to see you.

## 🎨 Guiding the Agent with Grammar

You can guide the `ToolCallingAgent` to produce tool calls in a specific format by using a **grammar**. The most common way is to use a regular expression to define the expected structure of the JSON output.

Here's how you can add a regex grammar to the previous example:

**Code Example (with grammar):**

    # ... (import statements, tool definition, model initialization) ...

    # Define a regex grammar for the tool-calling output
    json_grammar = {
        "type": "regex",
        "value": r'\{\s*"action":\s*"[\w_]+",\s*"action_input":\s*\{.*\}\s*\}'
    }

    # Initialize the ToolCallingAgent with the grammar
    tool_calling_agent = ToolCallingAgent(
        model=your_groq_model,
        tools=[greeting_tool],
        grammar=json_grammar,  # Pass the grammar here
        max_iterations=3
    )

    # ... (rest of the code) ...

**Explanation:**

*   **`json_grammar`:** This dictionary defines a regular expression that matches a simple JSON structure with an "action" key (the tool name) and an "action_input" key (the tool arguments).
*   **`grammar=json_grammar`:** We pass the grammar to the `ToolCallingAgent` during initialization.

Now, the agent will be guided to produce tool calls that conform to this JSON format, making parsing and execution more reliable.

## ✍️ Crafting the Perfect System Prompt

The system prompt is your primary way to instruct the `ToolCallingAgent` on its role, behavior, and how to use the available tools.

**Key Components of a System Prompt:**

1. **Agent's Persona:** Define the agent's role and capabilities (e.g., "You are a helpful assistant").
2. **Task Instructions:** Clearly state the overall goal of the agent and any specific instructions it should follow.
3. **Tool Usage:** Explain how the agent should use tools, including the expected output format (JSON in this case).
4. **`final_answer` Tool:** Emphasize that the agent should use the special `final_answer` tool to provide the final result.
5. **Placeholders:** Include these placeholders, which will be dynamically filled in by the `ToolCallingAgent`:
    *   **`{{tool_descriptions}}`:** This is where the descriptions of your tools will be inserted.
    *   **`{{tool_names}}`:** This is where the names of your tools will be inserted.

**Example System Prompt:**

    custom_system_prompt = """
    You are a helpful assistant. Your task is to assist the user by using the available tools.

    Follow these guidelines:

    - Only use the tools provided in the list: {{tool_names}}.
    - To use a tool, your output should be in the following JSON format:
        {
            "action": "tool_name",
            "action_input": {<tool arguments>}
        }
    - Provide your final answer using the `final_answer` tool.

    Available tools:

    {{tool_descriptions}}

    Let's get started!
    """

    tool_calling_agent = ToolCallingAgent(
        model=your_groq_model,
        tools=[greeting_tool],
        system_prompt=custom_system_prompt,
        # ... other arguments ...
    )

**Prompt Engineering Tips:**

*   **Be Explicit:** Leave no room for ambiguity in your instructions.
*   **Provide Examples:** If possible, include a few examples of valid tool calls in your system prompt.
*   **Iterate:** Experiment with different prompts to find what works best for your agent and task.
*   **Use Markdown:** You can use Markdown formatting within your system prompt to improve readability for the LLM (e.g., bolding, lists, etc.).

## 🧰 Advanced Tool Usage

### Creating More Complex Tools

Let's create a tool that performs a web search using the `DuckDuckGoSearchTool` and then summarizes the content of a web page using `visit_webpage` defined in previous `CodeAgent.md` file.

    from smolagents import DuckDuckGoSearchTool, tool
    from smolagents import ToolCallingAgent
    from smolagents import Tool
    import re
    import requests
    from markdownify import markdownify
    from requests.exceptions import RequestException
    from typing import Dict

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

    class WebSearchAndSummarizeTool(Tool):
        """
        Performs a web search using DuckDuckGo and summarizes the content of a given URL.
        """

        name = "web_search_and_summarize"
        description = (
            "This tool performs a web search using DuckDuckGo, retrieves the content of "
            "a specific result URL using the visit_webpage tool, and then summarizes the content."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "The search query for DuckDuckGo.",
            },
            "url": {
                "type": "string",
                "description": "The URL from the search results to summarize."
            }
        }
        output_type = "string"

        def __init__(self, model: LiteLLMModel, **kwargs):
            super().__init__(**kwargs)
            self.search_tool = DuckDuckGoSearchTool()
            self.summarizer_agent = ToolCallingAgent(
                model=model,
                tools=[visit_webpage],
                system_prompt="""
                    You are a helpful assistant that summarizes web page content.
                    Use the `visit_webpage` tool to get the page content, and then provide a concise summary.
                    Make sure to use the `final_answer` tool to provide the final summary.
                    {{tool_descriptions}}
                    """,
                max_iterations=3
            )

        def forward(self, query: str, url: str) -> str:
            """
            Performs a web search, retrieves and summarizes a web page.

            Args:
                query: The search query.
                url: The URL to summarize.

            Returns:
                A string containing the summary of the web page or an error message.
            """
            search_results = self.search_tool(query)

            # Find the result that matches the provided URL
            matched_result = None
            for result in search_results:
                if result['href'] == url:
                    matched_result = result
                    break

            if matched_result is None:
                return f"Error: Could not find the requested URL in the search results."

            # Use the summarizer agent to summarize the content of the URL
            summary = self.summarizer_agent.run(f"Summarize the content of this web page: {url}")
            return summary

**Explanation:**

*   **`WebSearchAndSummarizeTool`:**
    *   **`__init__`:** We initialize an instance of `DuckDuckGoSearchTool` for web searches and a `ToolCallingAgent` for summarizing.
    *   **`forward()`:**
        1. Performs a web search using `self.search_tool`.
        2. Finds the result that matches the provided URL.
        3. Uses a summarizer agent (`ToolCallingAgent`) to summarize the content of the URL.
        4. Returns the summary.

### Using the Advanced Tool

Now, let's integrate this new tool into our `ToolCallingAgent`:

    # ... (import statements, tool definitions) ...

    # Initialize the Groq model
    your_groq_model = LiteLLMModel(model_id="groq/llama3-70b-8192")

    # Create the WebSearchAndSummarizeTool instance
    web_search_and_summarize_tool = WebSearchAndSummarizeTool(model=your_groq_model)

    # Initialize the ToolCallingAgent with the new tool
    tool_calling_agent = ToolCallingAgent(
        model=your_groq_model,
        tools=[web_search_and_summarize_tool],
        max_iterations=5  # Increase if needed
    )

    # Run the agent
    task = "Find information about the latest advancements in AI and summarize the article from the most relevant search result."
    result = tool_calling_agent.run(task)
    print(result)

**Explanation:**

*   We create an instance of `WebSearchAndSummarizeTool`, passing in the `your_groq_model` for the summarizer agent to use.
*   We add `web_search_and_summarize_tool` to the `tools` list when initializing the `ToolCallingAgent`.
*   We give the agent a more complex task that requires both searching and summarizing.

## 🚀 Tips for Building Robust Agents

1. **Start Simple, Iterate:** Begin with a basic agent and a few tools. Gradually add complexity as you test and refine your system.
2. **Detailed Tool Descriptions:** The LLM relies heavily on tool descriptions. Make them as clear, concise, and informative as possible. Include details about input parameters, output format, and any limitations.
3. **Modular Design:** Break down complex tasks into smaller, manageable subtasks. This makes it easier for the agent to reason and plan. Consider using a multi-agent setup with `ManagedAgent` for this.
4. **Experiment with Prompts:** The system prompt and the way you phrase the task can significantly impact the agent's performance. Experiment with different prompt styles and task formulations.
5. **Model Choice:** Choose an LLM that is well-suited for your tasks and has good instruction-following and tool-use capabilities. Larger models often perform better but might be slower.
6. **Leverage Logs:** The `ToolCallingAgent` logs its steps. Use these logs to understand the agent's thought process, identify errors, and debug your system.

## 🚨 Error Handling and Debugging

*   **Tool Errors:** Implement **try-except** blocks within your tool's `forward()` method to catch potential errors and return informative error messages to the agent.
*   **Agent Errors:** Wrap the `agent.run()` call in a **try-except** block to handle any exceptions during the agent's execution.
*   **Logging:** Use Python's `logging` module to log important events, tool calls, and intermediate results. This can help you trace the agent's execution flow.
*   **Inspecting `agent.logs`:** After a run, inspect the `agent.logs` attribute to see a detailed record of the agent's steps, including the prompts sent to the LLM, the generated tool calls, and the observations.

## 🔮 Future Directions

*   **More Complex Grammars:** Explore using more advanced grammar formalisms (beyond regular expressions) to constrain the agent's output even further.
*   **Fine-tuning:** Consider fine-tuning an LLM specifically for tool calling with `ToolCallingAgent` to improve its performance on your specific tasks and tools.
*   **Integration with Other Frameworks:** Explore how to integrate `ToolCallingAgent` with other agent frameworks or orchestration systems.

## 🎯 Conclusion

The `ToolCallingAgent` provides a powerful and flexible way to build intelligent agents that can interact with the world through structured tool calls. By carefully designing your tools, crafting effective prompts, and leveraging the agent's reasoning capabilities, you can create sophisticated systems that automate complex tasks and unlock new possibilities in AI-driven applications. Remember to experiment, iterate, and refine your agent's design to achieve optimal performance!
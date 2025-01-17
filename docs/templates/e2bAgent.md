# 🚀 e2bAgent: Unleashing the Power of CodeAgent with Secure E2B Execution

The `e2bAgent.py` script demonstrates the power of combining `CodeAgent` from the `smolagents` library with the **E2B Code Interpreter** for secure and sandboxed code execution. This setup allows you to build intelligent agents that can generate and run Python code to perform complex tasks without compromising the security of your local system.

## 🌟 Why E2B is a Game-Changer for Code Agents

*   **🛡️ Enhanced Security:** E2B provides isolated cloud environments (sandboxes) where the code generated by your `CodeAgent` is executed. This is crucial when dealing with potentially untrusted code from an LLM, as it protects your local machine from malicious or erroneous code.
*   **⚙️ Controlled Environments:** Each E2B sandbox is a fresh, ephemeral environment. This reduces the risk of conflicts with your local setup and ensures consistent, reproducible execution.
*   **☁️ Scalability:** E2B sandboxes are cloud-based, so you can potentially scale your agent's operations more easily than if you were limited to local resources.

## 🛠️ Setting Up for E2B

1. **Install the E2B Package:**

    ```bash
    pip install e2b-code-interpreter
    ```

2. **Get an E2B API Key:**

    *   Sign up for an account on the [E2B website](https://e2b.dev/).
    *   Obtain an `API key` from your E2B dashboard.

3. **Set the `E2B_API_KEY` Environment Variable:**

    *   **Recommended:** Create a `.env` file in your project's root directory and add your API key:

        ```
        E2B_API_KEY=your_e2b_api_key
        ```

    *   **Alternative:** Set the environment variable directly in your terminal:

        ```bash
        export E2B_API_KEY="your_e2b_api_key"
        ```

## 🎯 Example: Data Analysis Agent with E2B

In this example, we'll create a `CodeAgent` that can perform basic data analysis on a CSV file. We'll define a custom tool, `DataAnalysisTool`, that leverages the `pandas` library to calculate statistics. The agent will then use this tool, along with the `PythonInterpreterTool`, to complete its task.

### 📄 `e2bAgent.py`

```python
import os
import pandas as pd
from smolagents import CodeAgent, LiteLLMModel, Tool, tool
from smolagents.default_tools import PythonInterpreterTool
from dotenv import load_dotenv

load_dotenv()

# Define the DataAnalysisTool
class DataAnalysisTool(Tool):
    name = "data_analyzer"
    description = "Loads a CSV file into a pandas DataFrame and provides basic analysis (e.g., mean, median, standard deviation) of a specified column. The filepath is relative to where you launched the script."
    inputs = {
        "filepath": {
            "type": "string",
            "description": "Path to the CSV file. This can be a file that you have uploaded to the session using additional_args.",
        },
        "column": {
            "type": "string",
            "description": "The name of the column to analyze.",
        },
    }
    output_type = "string"

    def forward(self, filepath: str, column: str) -> str:
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

# Initialize the Groq model using LiteLLM
your_groq_model = LiteLLMModel(model_id="groq/llama3-70b-8192")

# Create tool instances
data_analysis_tool = DataAnalysisTool()
python_interpreter_tool = PythonInterpreterTool(
    authorized_imports=["pandas", "dotenv"]
)

# Initialize the CodeAgent with E2B enabled
code_agent = CodeAgent(
    model=your_groq_model,
    tools=[python_interpreter_tool, data_analysis_tool],
    additional_authorized_imports=["pandas", "tempfile", "os"],
    use_e2b_executor=True,
)

# Example CSV data (replace with your actual data)
csv_data = """product,price,quantity
apple,1.0,10
banana,0.5,20
cherry,0.2,50
"""
with open("sales_data.csv", "w") as f:
    f.write(csv_data)

# Run the agent
try:
    result = code_agent.run(
        "Analyze the 'price' column in the dataset 'sales_data.csv' that I have uploaded to the session. Also, calculate the total value of all sales (price * quantity) and print it.",
        additional_args={"sales_data.csv": "sales_data.csv"}
    )
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")
```

#### Explanation:

- **DataAnalysisTool:** This custom tool takes a filepath and column name as input, reads the CSV using pandas, calculates the mean, median, and standard deviation, and returns a formatted string with the results.

- **PythonInterpreterTool:** This built-in tool allows the agent to execute general Python code.

### CodeAgent Initialization:

- **model:** We're using the Groq model via LiteLLMModel.

- **tools:** We provide both the data_analysis_tool and python_interpreter_tool.

- **additional_authorized_imports:** We include pandas (for the tool), tempfile, and os.

- **use_e2b_executor=True:** This is essential to enable E2B sandboxed execution.

- **additional_args:** We pass the path to the sales_data.csv as an additional argument. This makes the file accessible to the E2B environment.

## 🚀 Running the Example

- Save: Save the code above as e2bAgent.py.

- Run: Execute the script from your terminal:

```Bash
python3 e2bAgent.py
```

#### Expected Output:

The agent will generate code that uses the data_analysis_tool to analyze the sales_data.csv file and then likely use PythonInterpreterTool to calculate the total sales value. The output will be similar to this (the exact formatting might vary slightly):
```
... (Initialization logs) ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 analysis = data_analyzer(filepath='sales_data.csv', column='price')                │
│   2 print(analysis)                                                                     │
│   3                                                                                     │
│   4 df = pd.read_csv('sales_data.csv')                                                  │
│   5 total_sales = (df['price'] * df['quantity']).sum()                                  │
│   6 print(f"Total sales value: {total_sales}")                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
Execution logs:
Data Analysis for column 'price' in file 'sales_data.csv':
  Mean: 0.57
  Median: 0.50
  Standard Deviation: 0.40
Total sales value: 30.0

Out: None
[Step 0: Duration X.XX seconds| Input tokens: XXX | Output tokens: YY]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 final_answer(f"The average price is {analysis.split('Mean: ')[1].split('Median')[0].strip()}. The median price is {analysis.split('Median: ')[1].split('Standard Deviation')[0].strip()}. The standard deviation of the price is {analysis.split('Standard Deviation: ')[1].strip()}.\nThe total value of all sales is {total_sales}.")                                                                                                                                                          │

╰─────────────────────────────────────────────────────────────────────────────────────────╯
Out - Final answer: The average price is 0.57. The median price is 0.50. The standard deviation of the price is 0.40.
The total value of all sales is 30.0.
[Step 1: Duration X.XX seconds| Input tokens: XXX | Output tokens: YY]
The average price is 0.57. The median price is 0.50. The standard deviation of the price is 0.40.
The total value of all sales is 30.0.
```

## 🎨 Customizing the System Prompt

You can tailor the CodeAgent's behavior by modifying its `system_prompt`. When you initialize a `CodeAgent`, it uses a default system prompt that guides the underlying LLM. You can override this with your own.

**Important:** If you create a custom system prompt, ensure you include the following placeholders (using Jinja2 syntax), which the `CodeAgent` will dynamically populate:

- `{{tool_descriptions}}` Descriptions of the available tools.

- `{{tool_names}}` Names of the available tools.

- `{{managed_agents_descriptions}}` Descriptions of any managed agents (if you're using a multi-agent setup).

- `{{authorized_imports}}` The list of authorized Python imports.

#### Example:
```
custom_system_prompt = """
You are a powerful coding assistant designed to help with data analysis tasks.

Follow these guidelines:

-   Generate Python code that fulfills the user's request.
-   You **must** use the provided tools whenever appropriate.
-   If using `data_analyzer`, make sure the file exists.
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
    # ... other arguments ...
    system_prompt=custom_system_prompt
)
```

## 🚧 Error Handling and Logging

- **Tool Errors:** Implement try-except blocks within your tool's forward() method to catch potential errors (e.g., FileNotFoundError, KeyError, etc.) and return informative error messages.

- **Agent Errors:** Wrap the code_agent.run() call in a try-except block to handle any exceptions that might occur during the agent's execution.

- **Logs:** The CodeAgent automatically logs the generated code, tool calls, outputs, and errors. You can access these logs using the code_agent.logs attribute. Use these logs to debug and understand the agent's behavior.

## 🦺 E2B and Rate Limiting
### E2B

When using E2B, the code is executed in a separate, secure environment. It cannot directly access files on your local file system. You need to make use of the `additional_args` parameter in the `code_agent.run()` method to pass file names or other variables the agent might need.

### Rate Limiting

When using cloud-based LLMs like those from Groq, be mindful of the rate limits imposed by the API provider. The provided examples included basic rate limiting using `time.sleep()`. For production systems, you might need a more sophisticated rate-limiting strategy. You can use litellm's rate limiter for that.

## 🚀 Further Exploration

- Advanced Tool Usage: Explore creating more complex tools that interact with external APIs, databases, or other services.

- Multi-Agent Orchestration: Experiment with the ManagedAgent class to build multi-agent systems where different agents collaborate to solve a task.

- Prompt Engineering: Refine your system prompt and tool descriptions to improve the agent's performance.

- Gradio Interface: Use the GradioUI class to create a user-friendly web interface for your agent.

## Conclusion

This documentation provides a comprehensive and practical guide to using `CodeAgent` with `E2B` for secure code execution. Remember that building effective agents often involves experimentation and iteration. Use the provided examples as a starting point, and don't hesitate to explore and adapt them to create your own powerful AI coding assistants!

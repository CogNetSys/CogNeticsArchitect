import json
import os
from smolagents import ToolCallingAgent
from groq import Groq

# Define the GroqModel class to wrap the Groq Cloud API and provide the get_tool_call method
class GroqModel:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = os.getenv("GROQ_API_URL")
        if not self.api_key or not self.api_url:
            raise ValueError("GROQ_API_KEY and GROQ_API_URL must be set in environment variables.")
        self.client = Groq(api_key=self.api_key, base_url=self.api_url)

    # Basic call method for normal prompts
    def __call__(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            tool_choice="auto"
        )
        return response.choices[0].message["content"]

    # get_tool_call method to support ToolCallingAgent
    def get_tool_call(self, messages, tools):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )
        tool_call = response.choices[0].message.get('tool_calls', [None])[0]
        if tool_call:
            return {
                "name": tool_call['function']['name'],
                "arguments": json.loads(tool_call['function']['arguments'])
            }
        else:
            return None  # No tool call requested

# Define a sample tool function
def calculate(expression):
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Instantiate the GroqModel and ToolCallingAgent
groq_model = GroqModel()
groq_model.model = "llama3-groq-70b-tool-use"  # Replace with your desired Groq model
agent = ToolCallingAgent(model=groq_model, tools={"calculate": calculate})

# Example usage to test tool calling
if __name__ == "__main__":
    user_input = "What is 7 + 8?"
    response = agent.step(user_input)
    print("Agent's Response:", response)

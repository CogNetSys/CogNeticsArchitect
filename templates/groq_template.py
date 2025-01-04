from smolagents import CodeAgent, PythonInterpreterTool
import os
import requests


# Get the Groq API key and endpoint from environment variables
api_key = os.getenv("GROQ_API_KEY")
groq_endpoint = os.getenv("GROQ_API_ENDPOINT")

if api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set!")
if groq_endpoint is None:
    raise ValueError("GROQ_API_ENDPOINT environment variable not set!")


# Custom GroqModel class to interface with Groq Cloud
class GroqModel:
    def __init__(self, model_id, api_key, endpoint_url):
        self.model_id = model_id
        self.api_key = api_key
        self.endpoint_url = endpoint_url

    def __call__(self, messages, max_tokens=150, temperature=0.7, top_p=1.0, stop_sequences=None, **kwargs):
        # Remap or filter unsupported role values
        valid_roles = {"system", "user", "assistant"}
        cleaned_messages = [
            msg for msg in messages if msg["role"] in valid_roles
        ]

        # Construct payload
        payload = {
            "model": self.model_id,
            "messages": cleaned_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        # Send the request to Groq Cloud
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(self.endpoint_url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


# Instantiate the GroqModel
model = GroqModel(model_id="llama-3.1-8b-instant", api_key=api_key, endpoint_url=groq_endpoint)

# Add the Python Interpreter Tool to the agent
tools = [PythonInterpreterTool()]

# Create the CodeAgent with the model and tools
agent = CodeAgent(model=model, tools=tools)

# Run the agent with a math problem
response = agent.run("Calculate sin(987654321) + log10(123456789) using the math module.")
print("Assistant's Response:", response)

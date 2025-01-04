import os
from typing import List
import requests
from smolagents import CodeAgent
from transformers import tool

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
        """
        Sends messages to the Groq Cloud API and returns the assistant's response.
        """
        valid_roles = {"system", "user", "assistant"}
        cleaned_messages = [msg for msg in messages if msg["role"] in valid_roles]

        payload = {
            "model": self.model_id,
            "messages": cleaned_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(self.endpoint_url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

# Define the Wikipedia search tool
@tool
def search_wikipedia(query: str) -> List[str]:
    """
    Searches Wikipedia and returns a list of relevant page titles.

    Args:
        query (str): The search query to use on Wikipedia.
    """
    # Placeholder logic for demonstration
    return [f"Wikipedia Result for: {query}"]

# Instantiate the GroqModel
model = GroqModel(model_id="llama-3.1-8b-instant", api_key=api_key, endpoint_url=groq_endpoint)

# Create the CodeAgent with the search_wikipedia tool
agent = CodeAgent(model=model, tools=[search_wikipedia])

# Run the agent
response = agent.run("Who is Albert Einstein?")
print("Wikipedia Search Results:", response)

import os
import requests
from smolagents import CodeAgent, Tool
from transformers import tool  # Import the decorator
from huggingface_hub import HfApi

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

# Instantiate the GroqModel (replace with your actual model ID)
model = GroqModel(model_id="llama-3.1-8b-instant", api_key=api_key, endpoint_url=groq_endpoint)  

# Search tool using Hugging Face Hub API
@tool  # Apply the decorator
def search_hf_models(query: str, limit: int = 100) -> str:
    """
    Searches Hugging Face Hub for models based on a query.

    Args:
        query (str): The search query to use on the Hugging Face Hub.  This should be a string describing the type of model you are looking for (e.g., "text-classification").
        limit (int): The maximum number of models to return. Defaults to 100.

    Returns:
        str: A stringified list of model ids found on the Hugging Face Hub that match the query.  Returns an error message if the search fails.
    """
    api = HfApi()
    try:
        models = api.list_models(
            search_query=query,
            filter=("text-classification", "size_category=100M"),  # Keep the filter
            limit=limit
        )
        model_ids = [model.modelId for model in models]
        return str(model_ids)
    except Exception as e:
        return f"Error searching Hugging Face Hub: {e}"

model = GroqModel(model_id="llama-3.1-8b-instant", api_key=api_key, endpoint_url=groq_endpoint)

agent = CodeAgent(model=model, tools=[search_hf_models])

response = agent.run("List 100 free smaller size models for text classification from Hugging Face Hub.")
print("Free Models:", response)
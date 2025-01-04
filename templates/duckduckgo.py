import os
from smolagents import CodeAgent, DuckDuckGoSearchTool
from models import GroqModel


# --- Environment Variables and Model Initialization ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_ENDPOINT = os.environ.get("GROQ_API_ENDPOINT")


if not GROQ_API_KEY or not GROQ_API_ENDPOINT:
    raise ValueError("GROQ_API_KEY and GROQ_API_ENDPOINT environment variables must be set.")

groq_model = GroqModel(model_id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, endpoint_url=GROQ_API_ENDPOINT)


# --- Simple Search Test ---
tools = [DuckDuckGoSearchTool()]
agent = CodeAgent(model=groq_model, tools=tools, max_iterations=1)

try:
    response = agent.run("What is the capital of France?")
    print(f"\nSearch Agent Response:\n{response}")
except Exception as e:
    print(f"Error in Search Agent: {e}")
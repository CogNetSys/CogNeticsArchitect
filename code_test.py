import os
from templates.models import GroqModel
from templates.code_agent import GroqCodeAgentWithProperParsing, GroqCodeAgent
from smolagents import PythonInterpreterTool

# --- Environment Variables and Model Initialization ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_ENDPOINT = os.environ.get("GROQ_API_ENDPOINT")


if not GROQ_API_KEY or not GROQ_API_ENDPOINT:
    raise ValueError("GROQ_API_KEY and GROQ_API_ENDPOINT environment variables must be set.")

groq_model = GroqModel(model_id="groq/llama-3.1-8b-instant", api_key=GROQ_API_KEY)


# --- 1. Code Agent Test ---
tools = [PythonInterpreterTool()]
code_agent = GroqCodeAgentWithProperParsing(model=groq_model, tools=tools, max_iterations=1)
try:
    response = code_agent.run("What is today's date? Please format it as YYYY-MM-DD.")
    print(f"\nCode Agent Response (with proper parsing): {response}")
except Exception as e:
    print(f"Error in Code Agent (with proper parsing): {e}")

code_agent_no_parsing = GroqCodeAgent(model=groq_model, tools=tools, max_iterations=1)
try:
    response = code_agent_no_parsing.run("What is today's date? Please format it as YYYY-MM-DD.")
    print(f"\nCode Agent Response (no parsing): {response}")
except Exception as e:
    print(f"Error in Code Agent (no parsing): {e}")
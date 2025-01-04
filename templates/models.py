import os
import requests
from typing import List, Dict, Any
import time
from smolagents import LiteLLMModel
# --- Groq Model Integration ---

class GroqModel(LiteLLMModel):
    def __init__(self, model_id="groq/llama3-8b-8192", api_key=None, api_base=None, **kwargs):
        api_key = os.environ.get("GROQ_API_KEY")
        super().__init__(model_id=model_id, api_key=api_key, api_base=api_base, **kwargs)
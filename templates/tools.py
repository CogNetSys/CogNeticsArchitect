from smolagents import Tool
from typing import Optional

class MyTool(Tool):
    name = "summarizer"
    description = "Summarizes the given text"
    inputs = {"text": {"type": "string", "description": "The text to summarize"}}
    output_type = "string"

    def forward(self, text: str) -> str:
      """This is a dummy text summarizer"""
      return f"Summary of your provided text {text}, it was a pleasure summarizing it."
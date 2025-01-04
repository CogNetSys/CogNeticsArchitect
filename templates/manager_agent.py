from smolagents import CodeAgent, PythonInterpreterTool, Tool, ManagedAgent
from templates.models import GroqModel
from templates.tools import MyTool
import time
import re
from smolagents.utils import  AgentParsingError, AgentGenerationError, AgentExecutionError
from smolagents.agents import ToolCall
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.syntax import Syntax

class ManagerAgent(CodeAgent):
        def __init__(self, model, max_iterations = 1, **kwargs):
            super().__init__(tools=[], model=model, max_iterations=max_iterations, **kwargs)
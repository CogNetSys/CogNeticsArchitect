import os
from templates.models import GroqModel
from templates.code_agent import GroqCodeAgent, GroqCodeAgentWithProperParsing
from templates.tools import MyTool
from smolagents import PythonInterpreterTool, ManagedAgent

# --- Environment Variables and Model Initialization ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_ENDPOINT = os.environ.get("GROQ_API_ENDPOINT")


if not GROQ_API_KEY or not GROQ_API_ENDPOINT:
    raise ValueError("GROQ_API_KEY and GROQ_API_ENDPOINT environment variables must be set.")

groq_model = GroqModel(model_id="groq/llama3-8b-8192", api_key=GROQ_API_KEY)

# --- 3. Manager Agent ---
tools = [PythonInterpreterTool()]
code_agent = GroqCodeAgentWithProperParsing(model=groq_model, tools=tools, max_iterations=1)
my_summarizer = MyTool()

managed_code_agent = ManagedAgent(agent=code_agent, name="date_handler", description="Handles date related questions")
# This is the new code agent that does both summarizing and formatting.
managed_summarizer_code_agent = ManagedAgent(agent=GroqCodeAgent(model=groq_model, tools=[PythonInterpreterTool(), my_summarizer], max_iterations=1), name="summarizer_code", description="Summarizes the given text using python code.")

manager_agent = GroqCodeAgent(model=groq_model, tools=[], managed_agents=[managed_code_agent, managed_summarizer_code_agent], max_iterations=1)

try:
        manager_agent.system_prompt = """You are a manager agent that is responsible for delegating tasks to your managed agents.
          You are in charge of solving the task given by the user. To do so, you can leverage the following managed agents:
          {{managed_agents_descriptions}}

          Here are the rules you should always follow to solve your task:
          1. Always provide a 'Thought:' sequence, and a 'Action:' sequence, else you will fail.
          2. Use only variables that you have defined!
          3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
          4. You can't chain too many sequential tool calls in the same code block: make sure the output of a previous tool is printed out, and then use it in the next step.
          5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
          6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
          7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
          8. You can use imports in your code, and in particular the datetime module if you wish so.
          9. You are responsible for correctly formatting all the outputs.
          10. To call your managed agents, you should provide the text in argument to the agent's call: like 'date_handler(text="What is today\'s date?")' or 'summarizer_code(text="the text to summarize")' followed by a call to `final_answer()` with the results.
          11. Don't give up! You're in charge of solving the task, not providing directions to solve it.

        """
        response = manager_agent.run("Summarize this text: 'The very long text that no one could summarize' and provide me with today's date formatted as YYYY-MM-DD.")
        print(f"\nManager Agent Response: {response}")
except Exception as e:
    print(f"Error in Manager Agent: {e}")
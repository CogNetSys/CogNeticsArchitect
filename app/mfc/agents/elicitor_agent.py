import time
import json
import pandas as pd
from smolagents import CodeAgent, HfApiModel, Tool
from smolagents.default_tools import PythonInterpreterTool
from dotenv import load_dotenv
import re
import sys

# Add the CogNeticsArchitect root to the system path
sys.path.append("/home/irbsurfer/Projects/CogNeticsArchitect/")

# Import the NN function
from app.nn.cNN import process_structured_data  # Import the function

load_dotenv()

# Rate limit configuration
RATE_LIMIT_REQUESTS = 25  # requests allowed
RATE_LIMIT_PERIOD = 60  # seconds in which requests are limited
WAIT_INTERVAL = 15  # seconds to wait when limit is hit

# Cache for results to avoid redundant requests
response_cache = {}

# Track request counts
request_count = 0
last_reset_time = time.time()


# Rate limit function
def rate_limit():
    global request_count, last_reset_time
    current_time = time.time()
    if current_time - last_reset_time > RATE_LIMIT_PERIOD:
        request_count = 0
        last_reset_time = current_time
    if request_count >= RATE_LIMIT_REQUESTS:
        print(f"Rate limit exceeded. Waiting for {WAIT_INTERVAL}s.")
        time.sleep(WAIT_INTERVAL)
        request_count = 0
        last_reset_time = time.time()
    request_count += 1


# Retry logic and error handling
def run_with_retry(agent, prompt, additional_args=None, retries=3):
    for i in range(retries):
        try:
            # Apply rate limit
            rate_limit()

            cache_key = (prompt, str(additional_args))
            if cache_key in response_cache:
                print("Using cached response.")
                return response_cache[cache_key]

            response = agent.run(prompt, additional_args=additional_args)

            if not response:
                raise ValueError("Empty response from LLM.")

            response_cache[cache_key] = response
            return response

        except json.JSONDecodeError:
            print("Failed to decode JSON response. Skipping this attempt.")
            time.sleep(5)
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower():
                wait_time = WAIT_INTERVAL + i * 5  # Increase wait time with retries
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}")
    raise Exception("Max retries reached.")


# Define tools
class DataExtractorTool(Tool):
    name = "data_extractor"
    description = "Extracts structured data (goal, metric, constraints, timeline) from input text."
    inputs = {
        "input_text": {
            "type": "string",
            "description": "Unstructured text input for goal directives or feedback.",
        }
    }
    output_type = "string"

    def forward(self, input_text: str) -> str:
        try:
            goal_match = re.search(
                r"(increase|improve|reduce|achieve)\s(.+?)\b", input_text, re.IGNORECASE
            )
            budget_match = re.search(r"\$\d+[kKmM]", input_text)
            timeline_match = re.search(r"(Q[1-4]\s\d{4}|\b\d{4}\b)", input_text)
            metric_match = re.search(r"\d+%", input_text)

            extracted_data = {
                "goal": goal_match.group(0) if goal_match else "N/A",
                "constraint": (
                    f"Budget cap: {budget_match.group(0)}" if budget_match else "None"
                ),
                "timeline": timeline_match.group(0) if timeline_match else "N/A",
                "metric": metric_match.group(0) if metric_match else "N/A",
            }
            return json.dumps(extracted_data)

        except Exception as e:
            return f"Error during extraction: {e}"


class ClarificationTool(Tool):
    name = "clarification_agent"
    description = "Clarifies missing key elements (goal, metric, constraint, timeline) using user interaction or default LLM guidance."
    inputs = {
        "missing_key": {
            "type": "string",
            "description": "The missing element to clarify (e.g., goal, metric).",
        },
        "input_text": {
            "type": "string",
            "description": "The original directive or feedback text.",
        },
    }
    output_type = "string"

    def forward(self, missing_key: str, input_text: str) -> str:
        if missing_key.lower() == "metric":
            return input(
                f"Clarification needed: What is the desired percentage increase/decrease for {missing_key} (e.g., 'increase by 15%')? "
            )
        elif missing_key.lower() == "goal":
            return input(
                f"Clarification needed: Can you specify the goal more clearly for '{input_text}'? "
            )
        else:
            return f"Clarifying {missing_key} from '{input_text}'..."


# Model initialization
hf_model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Tool instances
data_extractor_tool = DataExtractorTool()
clarification_tool = ClarificationTool()

# Code agent with all tools
code_agent = CodeAgent(
    model=hf_model,
    tools=[data_extractor_tool, clarification_tool],
    additional_authorized_imports=["re", "json"],
)


# Elicit structured data and pass directly to NN
def elicit_structured_data(input_text):
    print("\n--- Running Data Extraction ---")
    try:
        # Extract structured data
        extraction_result = run_with_retry(
            code_agent,
            "Use the data_extractor tool to extract structured data from the input text.",
            additional_args={"input_text": input_text},
        )

        # Debug: Print the raw extraction result
        print(f"Raw Extraction Result: {extraction_result}")

        # Ensure extraction result is a valid JSON string or dict
        if isinstance(extraction_result, str):
            try:
                structured_data = json.loads(
                    extraction_result.replace("'", '"')
                )  # Convert single quotes to double quotes for JSON
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON format in extraction result: {e}\nResult: {extraction_result}"
                )
        elif isinstance(extraction_result, dict):
            structured_data = extraction_result
        else:
            raise ValueError(
                f"Unexpected extraction result type: {type(extraction_result)}"
            )

        # Check if structured data is valid
        if not isinstance(structured_data, dict):
            raise ValueError(f"Structured data is not a dictionary: {structured_data}")

        print(
            f"\n--- Extracted Structured Data ---\n{json.dumps(structured_data, indent=4)}"
        )

        # Clarification for missing fields
        for key, value in structured_data.items():
            if not value or value.lower() == "n/a":
                clarification = run_with_retry(
                    code_agent,
                    "Use the clarification_agent tool to ask for missing information.",
                    additional_args={"missing_key": key, "input_text": input_text},
                )
                structured_data[key] = clarification

        print(
            f"\n--- Final Structured Data ---\n{json.dumps(structured_data, indent=4)}"
        )

        # Pass structured data to the NN
        process_structured_data(structured_data)

    except Exception as e:
        print(f"Error during structured data elicitation: {e}")


# Example usage
if __name__ == "__main__":
    input_text = "We need to increase customer retention by 15% while staying under $500K in Q1 2025."
    elicit_structured_data(input_text)

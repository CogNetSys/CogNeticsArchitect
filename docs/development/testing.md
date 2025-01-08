# Testing Guide
## Running Tests
To run all tests:
pytest

## Writing New Tests
New tests should be added in the `tests/` folder.

## Example Test
import pytest
from app.CodeAgent import CodeAgent

def test_refactor_code():
    code_agent = CodeAgent()
    assert code_agent.refactor_code("def old(): pass") == "def new_function(): pass"

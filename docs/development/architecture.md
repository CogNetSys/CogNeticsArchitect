# System Architecture
The CogNetics Architect system follows a modular, multi-agent architecture:
- **CodeAgent**: Responsible for handling and updating code snippets.
- **ToolAgent**: Manages external tool usage.
- **WebSearchAgent**: Handles web-based searches and integrates search results.

## System Components
1. **App**: Core logic and API handling.
2. **Docs**: MkDocs-based documentation.
3. **Tests**: Unit tests for code validation.

/app
|-- CodeAgent.py
|-- ToolAgent.py
|-- WebSearchAgent.py

*(Add a diagram here if available)*

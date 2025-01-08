# Project Setup Guide

## 1. Install Essential Tools

### System Preparation
- **Install Python**: Download and install the latest version of Python from [python.org](https://www.python.org).
- **VS Code**: Download from [code.visualstudio.com](https://code.visualstudio.com).

### VS Code Extensions:
1. **Python** (by Microsoft)
2. **Pylance** (for intelligent type-checking)
3. **Jupyter** (for notebooks)
4. **GitHub Copilot** (optional for AI assistance)
5. **GitLens** (to enhance Git experience)
6. **Prettier** (for consistent code formatting)
7. **Markdown All in One** (for enhanced README editing)
8. **dotenv** (for `.env` file support)
9. **EditorConfig** (for uniform settings across editors)

## 2. Folder Structure
```
/my_python_project
|-- .github/ (for GitHub Actions workflows)
|-- .vscode/
|   |-- settings.json (VS Code project-specific settings)
|   |-- launch.json (debug configuration)
|-- src/ (source code)
|   |-- __init__.py
|-- tests/ (unit tests)
|-- docs/ (documentation)
|-- .gitignore
|-- .env (for environment variables)
|-- requirements.txt (or `pyproject.toml` for modern dependency management)
|-- README.md
|-- LICENSE
```

## 3. Initialize Git and Connect to GitHub
- Open **VS Code Terminal**.
- Initialize Git repository:
  ```
  git init
  ```
- Create a new GitHub repository and add the remote:
  ```
  git remote add origin <your-repo-url>
  ```

## 4. Python Virtual Environment
- Create a virtual environment:
  ```
  python -m venv venv
  ```
- Activate the environment:
  - **Windows**:
    ```
    .\\venv\\Scripts\\activate
    ```
  - **Mac/Linux**:
    ```
    source venv/bin/activate
    ```

## 5. Install Dependencies
- Create `requirements.txt`:
  ```
  pip freeze > requirements.txt
  ```

## 6. VS Code Configuration
- **settings.json** (`.vscode/settings.json`):
  ```
  {
    "python.pythonPath": "venv/bin/python",
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python",
    "files.exclude": {
      "**/__pycache__": true,
      "**/*.pyc": true,
      "**/.DS_Store": true
    },
    "python.linting.enabled": true,
    "python.languageServer": "Pylance"
  }
  ```

## 7. Create .gitignore
```
# Python
__pycache__/
*.pyc
*.pyo
venv/
.vscode/
.env
logs/
```

## 8. Add Testing Framework
- Install `pytest`:
  ```
  pip install pytest
  ```

## 9. Add Linter and Formatter
- **Black** (formatter):
  ```
  pip install black
  ```
- **Pylint** (linter):
  ```
  pip install pylint
  ```

## 10. Pre-Commit Hooks (Optional)
- Install `pre-commit`:
  ```
  pip install pre-commit
  ```
- Create `.pre-commit-config.yaml`:
  ```
  repos:
    - repo: https://github.com/psf/black
      hooks:
        - id: black
  ```
- Run:
  ```
  pre-commit install
  ```

## 11. Add GitHub Actions for CI/CD
- Create `.github/workflows/python.yml`:
  ```
  name: Python CI
  on:
    push:
      branches:
        - main
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
        - name: Install dependencies
          run: pip install -r requirements.txt
        - name: Run tests
          run: pytest
  ```

## 12. Environment Variables Setup
- Create a `.env` file:
  ```
  DEBUG=True
  SECRET_KEY=supersecretkey
  ```

## 13. Code Documentation
- Install `mkdocs`:
  ```
  pip install mkdocs
  ```
- Run locally:
  ```
  mkdocs serve
  ```

## 14. Push to GitHub
```
git add .
git commit -m "Initial commit"
git push -u origin main
```

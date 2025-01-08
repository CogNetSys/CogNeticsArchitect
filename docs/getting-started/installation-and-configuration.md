# Installation and Configuration Guide 🛠️

Welcome, contributor! 🎉 This guide will help you set up and configure the **CogNetics Architect** project for local development.

## 1. Clone the Repository 🖥️

First, fork the repository from GitHub and clone it locally:
```bash
git clone https://github.com/your-username/CogNeticsArchitect.git
cd CogNeticsArchitect
```

## 2. Create and Activate the Virtual Environment 🌐

We use a virtual environment named `CogNeticsArchitect` to manage dependencies:
```bash
python -m venv CogNeticsArchitect
```

Activate the virtual environment:
- **Windows**:
  ```bash
  .\\CogNeticsArchitect\\Scripts\\activate
  ```
- **Mac/Linux**:
  ```bash
  source CogNeticsArchitect/bin/activate
  ```

## 3. Install Dependencies 📦

Install the project dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

If you're using `poetry`:
```bash
poetry install
```

## 4. Configure Environment Variables 📝

Add a `.env` file in the project root to store environment variables:
```
DEBUG=True
DATABASE_URL=postgres://username:password@localhost/dbname
SECRET_KEY=supersecretkey
```

## 5. Verify Installation ✅

To ensure everything works, run the following:
```bash
pytest
```

You should see all tests passing! 🎉

## 6. Git Configuration 🌐

Set your Git username and email:
```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

## 7. Optional Pre-Commit Hook Setup 🧹

We use pre-commit hooks to maintain clean and consistent code:
```bash
pip install pre-commit
pre-commit install
```

This will run `black` and lint checks before every commit.

## 8. Running MkDocs Locally 📝

To preview the documentation:
```bash
mkdocs serve --dev-addr 0.0.0.0:7492
```

Visit `http://localhost:7492` to view the docs.

## You're All Set! 🚀

You’re now ready to contribute! Make sure to follow the [Contributing Guide](../contributing.md) for submitting your pull requests.

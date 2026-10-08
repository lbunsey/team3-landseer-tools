# Landseer Agent Tools

A LangChain-based agent designed to help users manage models and datasets in a repository through natural language interactions.

## Setup

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

**Windows (PowerShell):**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Open `.env` and configure the LLM provider and its corresponding settings according to your needs.

For example:

```env
LLM_PROVIDER=google

GOOGLE_MODEL=your-model-name
GOOGLE_API_KEY=your-api-key
```

Available providers and their required environment variables are documented in `.env.example`.

**Important:** Do not commit your `.env` file or expose your API keys.

### 4. Run the Agent

After completing the setup, run:

```bash
python agent.py
```

To end the conversation, type `exit` or `quit`.

## Project Structure

```text
Landseer Agent Tools/
├── agent.py
├── tools/
│   ├── __init__.py
│   ├── clone_repo.py
│   ├── list_datasets.py
|   └── ... 
├── repos/
├── .env.example
├── requirements.txt
└── README.md
```

The `tools/` directory contains the tools available to the agent.

`tools/__init__.py` serves as the central tool registry. When adding a new tool, add its import and include it in the `TOOLS` list in `tools/__init__.py`.

For example:

```python
from .clone_repo import clone_repo
from .list_datasets import list_datasets
from .new_tool import new_tool

TOOLS = [
    clone_repo,
    list_datasets,
    new_tool,
]
```


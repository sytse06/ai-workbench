# AI workbench 

This Langchain based AI workbench enables users to interact with various language models (LLMs) through a user-friendly interface. Users can engage in real-time conversations, select from multiple models, and reuse prompts for efficient task completion, making it a versatile tool for both casual and professional use.

## Setup

- Ensure you have Poetry installed.
- Clone this repository.
- Navigate to the project directory.

### Project structure:
```
langchain working bench/
├── ai_model_core/
│   ├── __init__.py
│   ├── factory.py
│   ├── utils.py
│   ├── model_helpers/
│   │   ├── __init__.py
│   │   ├── chat_assistant_.py
│   │   ├── prompt_assistant.py
│   │   ├── RAG_assistant.py
│   │   └── vision_assistant.py
│   └── config/
│       ├── __init__.py
│       ├── credentials.py
│       ├── settings.py
│       └── config.yaml
├── input/
├── output/
├── tests/
└── main.py
```
### Component flow ai_model_interface
Contents __init.py
from .factory import get_model
from .config.credentials import load_credentials, get_api_key
from .utils import format_prompt, get_system_prompt, get_prompt_template, format_history
from .config.settings import load_config, get_directory, get_prompt, get_prompt_list, update_prompt_list
```
utils.py
    ↓
__init__.py
    ↓
factory.py
    ↓
main.py
```

## Installation

- Run `poetry install` to set up the virtual environment and install dependencies.
- Ensure Ollama is installed and the Llama3.1 and LLaVA models are pulled (`ollama pull [model_name]`).

## Usage

- Activate the virtual environment: `poetry shell`
- Copy/add the image files with the notes to the input directory.
- Run the script: `python main.py`

The script will spin up a browser tab where the Gradio UI serves the available model assistants of the AI workbench.
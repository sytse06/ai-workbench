# AI workbench 

This Langchain based AI workbench enables users to interact with various language models (LLMs) through a browser based interface. Users can engage in real-time conversations, select from multiple models, and reuse prompts for efficient task completion, making it a versatile tool to streamline complex tasks and foster rapid experimentation.

## Setup

- Ensure you have Poetry installed.
- Clone this repository.
- Navigate to the project directory.

### Project structure:
```
ai_workbench/
├── ai_model_core/
│   ├── __init__.py
│   ├── shared_utils/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── utils.py
│   │   ├── message_processing.py
│   │   └── prompt_utils.py
│   ├── model_helpers/
│   │   ├── __init__.py
│   │   ├── chat_assistant.py
│   │   ├── RAG_assistant.py
│   │   ├── summarize_assistant.py
│   │   ├── transcription_assistant.py
│   └── config/
│       ├── __init__.py
│       ├── credentials.py
│       ├── settings.py
│       └── config.yaml
├── input/
├── output/
├── tests/
│   ├── __init__.py
│   └── chat_assistant/
│       └── test_chat_assistant.py
├── .env
└── main.py
```
### Features

Real-time chat interface
Multiple model support (Ollama, Claude, Gemini, Openai, etc.)
File upload and processing
Audio transcription
Document summarization
RAG capabilities
Vision processing
Customizable prompts
Conversation history management

### Dependencies
LangChain
Gradio
PyTorch
Transformers
Ollama

### Component flow ai_model_interface
Contents __init.py
from .factory import get_model, get_embedding_model
from .config.credentials import load_credentials, get_api_key
from .shared_utils.utils import format_prompt, get_system_prompt, get_prompt_template, format_history
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
- Save your api_keys for paid model use in a credentials.json file and specify its location in CREDENTIALS_PATH variable in credentials.py.
- Run the script: `python main.py`

The script will spin up a browser tab where the Gradio UI serves the available model assistants of the AI workbench.
# Langchain working bench project

This project converts handwritten notes to digital text using LLaVA and Ollama.

## Setup

- Ensure you have Poetry installed.
- Clone this repository.
- Navigate to the project directory.

### Project structure:
```
langchain working bench/
├── ai_model_interface/
│   ├── __init__.py
│   ├── factory.py
│   ├── utils.py
│   ├── model_helpers/
│   │   ├── __init__.py
│   │   ├── chatbot.py
│   │   ├── transcription_assistant.py
│   │   ├── rag_assistant.py
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
from .utils import format_prompt, get_system_prompt, get_prompt_template, format_history
from .config.credentials import load_credentials, get_api_key
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
- Run the script: `python note_converter.py`

The script will process all images in the input directory and save the results as markdown files in the output directory.
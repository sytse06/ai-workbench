# Notetaking Conversion Project

This project converts handwritten notes to digital text using LLaVA and Ollama.

## Setup

- Ensure you have Poetry installed.
- Clone this repository.
- Navigate to the project directory.

### Project structure:
```
    notetaking-conversion/
    ├── pyproject.toml
    ├── config.yaml
    ├── notetaking_conversion/
    │   ├── __init__.py
    │   └── note_converter.py
    ├── tests/
    │   └── test_note_converter.py
    ├── input/
    └── output/
```

- Run `poetry install` to set up the virtual environment and install dependencies.
- Ensure Ollama is installed and the LLaVA model is pulled (`ollama pull llava`).

## Usage

- Activate the virtual environment: `poetry shell`
- Copy/add the image files with the notes to the input directory.
- Run the script: `python note_converter.py`

The script will process all images in the input directory and save the results as markdown files in the output directory.
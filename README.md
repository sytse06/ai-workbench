# Notetaking Conversion Project

This project converts handwritten notes to digital text using LLaVA and Ollama.

## Setup

1. Ensure you have Poetry installed.
2. Clone this repository.
3. Navigate to the project directory.
4. Project structure:
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
5. Run `poetry install` to set up the virtual environment and install dependencies.
6. Ensure Ollama is installed and the LLaVA model is pulled (`ollama pull llava`).

## Usage

1. Activate the virtual environment: `poetry shell`
2. Edit the `note_converter.py` file to set the correct input and output directories.
3. Run the script: `python note_converter.py`

The script will process all images in the input directory and save the results as markdown files in the output directory.
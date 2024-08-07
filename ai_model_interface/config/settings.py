# ai_model_interface/config/settings.py
import os
import yaml
import logging
from typing import List
import gradio as gr

logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'system' not in config:
            raise KeyError("Config file is missing 'system' section")
        
        if 'directories' not in config['system']:
            raise KeyError("Config file is missing 'directories' section")
        
        if 'prompts' not in config:
            raise KeyError("Config file is missing 'prompts' section")
        
        logger.info("Config loaded successfully")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ValueError(f"Error parsing YAML file: {e}")

def get_directory(dir_type: str) -> str:
    config = load_config()
    return config['system']['directories'].get(dir_type)

def get_prompt(prompt_name: str) -> str:
    config = load_config()
    return config['prompts'].get(prompt_name)

def get_prompt_list(language: str) -> List[str]:
    config = load_config()
    prompts = config.get("prompts", {})
    return prompts.get(language, [])

# Function to update prompt list based on language choice
def update_prompt_list(language: str):
    new_prompts = get_prompt_list(language)
    return gr.Dropdown.update(choices=get_prompt_list(language))
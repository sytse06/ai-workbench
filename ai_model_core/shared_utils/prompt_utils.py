# ai_model_core/shared_utils/prompt_utils.py
# Standard library imports
from typing import List, Dict
import logging

# Third-party imports
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ..config.settings import load_config

def get_system_prompt(language_choice: str, config: dict) -> str:
    try:
        return config["system_prompt_settings"][language_choice]["system_prompt"]
    except KeyError:
        logger.error(f"System prompt not found for language: {language_choice}")
        return "Default system prompt"

def get_prompt_template(prompt_info: str, config: dict, language_choice: str = "english") -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate using the prompt_info and config.

    :param prompt_info: The prompt info selected by the user
    :param config: The loaded configuration
    :return: ChatPromptTemplate
    """
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    prompt_text = config['prompts'].get(prompt_info, "")
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{prompt_info}\n\n{user_message}")
    ])
def get_prompt(prompt_name: str) -> str:
    config = load_config()
    return config['prompts'].get(prompt_name)

def get_prompt_list(language: str) -> List[str]:
    config = load_config()
    prompts = config.get("prompts", {})
    return prompts.get(language, [])

def update_prompt_list(language: str):
    new_prompts = get_prompt_list(language)
    return gr.Dropdown(choices=new_prompts)
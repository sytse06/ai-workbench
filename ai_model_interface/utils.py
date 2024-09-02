# ai_model_interface/utils.py
from langchain.prompts import ChatPromptTemplate
from typing import List, Union, Any
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
import logging

logger = logging.getLogger(__name__)

def get_system_prompt(language_choice: str, config: dict) -> str:
    try:
        return config["system_prompt_settings"][language_choice]["system_prompt"]
    except KeyError:
        logger.error(f"System prompt not found for language: {language_choice}")
        return "Default system prompt"

def get_prompt_template(prompt_info: str, config: dict) -> ChatPromptTemplate:
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

def _format_history(self, history: List[tuple[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history
# ai_model_interface/utils.py
from langchain.prompts import ChatPromptTemplate

def format_prompt(system_prompt: str, user_message: str, prompt_info: str) -> ChatPromptTemplate:
    """
    Formats the prompt using Langchain's PromptTemplate.

    :param system_prompt: The system message
    :param user_message: The user's message
    :param prompt_info: Additional prompt information
    :return: Formatted prompt string
    """
    prompt_template = ChatPromptTemplate.from_template("{system_prompt}\n\n{prompt_info}\n\n{user_message}")
    return prompt_template
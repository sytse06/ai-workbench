# ai_model_interface/utils.py
from langchain.prompts import PromptTemplate

def format_prompt(system_prompt: str, user_message: str, prompt_info: str) -> str:
    """
    Formats the prompt using Langchain's PromptTemplate.

    :param system_prompt: The system message
    :param user_message: The user's message
    :param prompt_info: Additional prompt information
    :return: Formatted prompt string
    """
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "user_message", "prompt_info"],
        template="{system_prompt}\n\n{prompt_info}\n\n{user_message}"
    )
    return prompt_template.format(
        system_prompt=system_prompt,
        user_message=user_message,
        prompt_info=prompt_info
    )
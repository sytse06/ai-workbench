# ai_model_interface/factory.py
from .models.ollama import OllamaModel
from .models.openai import OpenAIModel
from .models.anthropic import AnthropicModel
from ai_model_interface.config.credentials import get_api_key
from langchain.prompts import PromptTemplate

def get_model(choice: str, **kwargs):
    if choice == "Ollama (LLama3.1)":
        return OllamaModel(model_name="llama3.1", **kwargs)
    elif choice == "Ollama (LLaVA)":
        return OllamaModel(model_name="llava", **kwargs)
    elif choice == "Ollama (Deepseek-coder-v2)":
        return OllamaModel(model_name="deepseek-coder-v2", **kwargs)
    elif choice == "OpenAI GPT-4o-mini":
        api_key = get_api_key('openai')
        return OpenAIModel(model_name="gpt-4o-mini", api_key=api_key, **kwargs)
    elif choice == "Anthropic Claude":
        api_key = get_api_key('anthropic')
        return AnthropicModel(model_name="claude-3-sonnet-20240229", api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Invalid model choice: {choice}")

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
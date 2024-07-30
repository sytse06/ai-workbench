# ai_model_interface/factory.py
from .models.ollama import OllamaModel
from .models.openai import OpenAIModel
from .models.anthropic import AnthropicModel

def get_model(choice: str, **kwargs):
    if choice == "Ollama (LLaVA)":
        return OllamaModel(model_name="llava", **kwargs)
    elif choice == "OpenAI GPT-4o-mini":
        api_key = get_api_key('openai')
        return OpenAIModel(model_name="gpt-4o-mini", api_key=api_key, **kwargs)
    elif choice == "Anthropic Claude":
        api_key = get_api_key('anthropic')
        return AnthropicModel(model_name="claude-3-sonnet-20240229", api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Invalid model choice: {choice}")
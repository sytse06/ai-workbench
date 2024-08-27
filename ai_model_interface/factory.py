# ai_model_interface/factory.py
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from ai_model_interface.config.credentials import get_api_key
from langchain.prompts import PromptTemplate

def get_model(choice: str, **kwargs):
    if choice == "Ollama (LLama3.1)":
        return ChatOllama(base_url="http://localhost:11434", model_name="llama3.1", **kwargs)
    elif choice == "Ollama (LLaVA)":
        return ChatOllama(base_url="http://localhost:11434", model_name="llava", **kwargs)
    elif choice == "Ollama (Deepseek-coder-v2)":
        return ChatOllama(base_url="http://localhost:11434", model_name="deepseek-coder-v2", **kwargs)
    elif choice == "OpenAI GPT-4o-mini":
        api_key = get_api_key('openai')
        return ChatOpenAI(model_name="gpt-4o-mini", api_key=api_key, **kwargs)
    elif choice == "Anthropic Claude":
        api_key = get_api_key('anthropic')
        return ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Invalid model choice: {choice}")
# ai_model_interface/factory.py
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from ai_model_interface.config.credentials import get_api_key
from langchain.prompts import PromptTemplate

def get_model(choice: str, **kwargs):
    if choice == "Ollama (LLama3.1)":
        return ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            verbose=True)
    elif  choice == "Ollama (phi3.5)":
        return ChatOllama(
            model="phi3.5",
            base_url="http://localhost:11434",
            verbose=True)
    elif choice == "Ollama (LLaVA)":
        return ChatOllama(
            model="llava",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (Deepseek-coder-v2)":
        return ChatOllama( 
            model="deepseek-coder-v2",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "OpenAI GPT-4o-mini":
        api_key = get_api_key('openai')
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=api_key, 
            **kwargs)
    elif choice == "Anthropic Claude":
        api_key = get_api_key('anthropic')
        return ChatAnthropic(
            model="claude-3-sonnet-20240229", 
            api_key=api_key, 
            **kwargs)
    else:
        raise ValueError(f"Invalid model choice: {choice}")
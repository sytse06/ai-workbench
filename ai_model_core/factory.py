# ai_model_interface/factory.py
import openai
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama, ChatAnthropic
from langchain_openai import ChatOpenAI
from ai_model_core.config.credentials import get_api_key, load_credentials
from langchain.prompts import PromptTemplate

def get_model(choice: str, **kwargs):
    load_credentials() 
    
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
    elif choice == "Ollama (llava:7b-v1.6)":
        return ChatOllama(
            model="llava:7b-v1.6",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (Deepseek-coder-v2)":
        return ChatOllama( 
            model="deepseek-coder-v2",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (YI-coder)":
        return ChatOllama( 
            model="yi-coder",
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
    
def get_embedding_model(choice: str, **kwargs):
    load_credentials()
    
    if choice == "nomic-embed-text":
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            **kwargs
        )
    elif choice == "all-MiniLM-L6-v2":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            **kwargs
        )
    elif choice == "text-embedding-ada-002":
        api_key = get_api_key('openai')
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=api_key,
            **kwargs
        )
    elif choice == "text-embedding-3-small":
        api_key = get_api_key('openai')
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
            **kwargs
        )
    elif choice == "text-embedding-3-large":
        api_key = get_api_key('openai')
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid embedding model choice: {choice}")
# Third-party imports
import whisper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from ..config.credentials import get_api_key, load_credentials

__all__ = ['get_model', 'get_embedding_model', 'WHISPER_MODELS', 'OUTPUT_FORMATS']

# Whisper model options
WHISPER_SIZES = {
    "tiny": "tiny",
    "base": "base", 
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "large-v2", 
    "large-v3": "large-v3"
}
OUTPUT_FORMATS = ["none", "txt", "srt", "vtt", "tsv", "json", "all"]
WHISPER_MODELS = [f"Whisper {size}" for size in WHISPER_SIZES.keys()]


def get_model(choice: str, **kwargs):
    """
    Create a language model instance for inference with LLM API
    
    Args:
        choice (str): Choice of model to use 
            (e.g. "Ollama (LLama3.1)")
        **kwargs: Additional keyword arguments for model configuration
    
    Returns:
        ChatModel: Instance of the selected chat model
    """
    load_credentials() 
    
    if choice == "Ollama (LLama3.1)":
        return ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            verbose=True)
    elif choice == "Claude Sonnet":
        api_key = get_api_key('openrouter')
        model_kwargs = {}
        if 'http_referer' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["HTTP-Referer"] = kwargs.pop('http_referer')
        if 'x_title' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["X-Title"] = kwargs.pop('ai-workbench')
        
        return ChatOpenAI(
            model="anthropic/claude-3.5-sonnet",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=model_kwargs,
            **kwargs
        )
    elif choice == "Deepseek v3":
        api_key = get_api_key('openrouter')
        model_kwargs = {}
        if 'http_referer' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["HTTP-Referer"] = kwargs.pop('http_referer')
        if 'x_title' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["X-Title"] = kwargs.pop('ai-workbench')
        
        return ChatOpenAI(
            model="deepseek/deepseek-chat",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=model_kwargs,
            **kwargs
        )
    elif choice == "Claude Sonnet beta":
        api_key = get_api_key('openrouter')
        model_kwargs = {}
        if 'http_referer' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["HTTP-Referer"] = kwargs.pop('http_referer')
        if 'x_title' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["X-Title"] = kwargs.pop('ai-workbench')
        
        return ChatOpenAI(
            model="anthropic/claude-3.5-sonnet:beta",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs=model_kwargs,
            **kwargs
        )
    elif choice == "Gemini 1.5 flash":
        api_key = get_api_key('google')
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            convert_system_message_to_human=True,
            **kwargs
        )
    elif choice == "Ollama (LLama3.2)":
        return ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (llama3.2-vision)":
        return ChatOllama(
            model="llama3.2-vision",
            base_url="http://localhost:11434",
            **kwargs)    
    elif choice.startswith("Whisper"):
        whisper_size = choice.split()[-1].lower()
        if whisper_size not in WHISPER_SIZES:
            raise ValueError(
                f"Invalid Whisper model size '{whisper_size}'. Choose from: {', '.join(WHISPER_SIZES.keys())}"
            )
        try:
            return whisper.load_model(WHISPER_SIZES[whisper_size])
        except Exception as e:
            raise ValueError(f"Failed to load Whisper model: {str(e)}")
    elif choice == "Mistral (large)":
        api_key = get_api_key('mistral')
        return ChatOpenAI(
            model="mistral-large-latest",
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            **kwargs
        )
    elif choice == "Mistral (pixtral)":
        api_key = get_api_key('mistral')
        return ChatOpenAI(
            model="pixtral-12b-2409",
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            **kwargs
        )
    elif choice == "Mistral (small)":
        api_key = get_api_key('mistral')
        return ChatOpenAI(
            model="mistral-small-latest",
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            **kwargs
        )
    elif choice == "Ollama (phi3.5)":
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
    elif choice == "bge-large":
        return OllamaEmbeddings(
            model="bge-large",
            base_url="http://localhost:11434",
            **kwargs
        )
    elif choice == "bge-m3":
        return OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434",
            **kwargs
        )
    if choice == "e5-large":
        # Import E5Embeddings locally to avoid circular import
        from ..model_helpers.embeddings import E5Embeddings
        return E5Embeddings(
            model_name="intfloat/multilingual-e5-large",
            **kwargs
        )
    if choice == "e5-base":
        # Import E5Embeddings locally to avoid circular import
        from ..model_helpers.embeddings import E5Embeddings
        return E5Embeddings(
            model_name="intfloat/multilingual-e5-base",
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

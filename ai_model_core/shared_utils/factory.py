# ai_model_core/shared_utils/factory.py
# Standard library imports
import os
import sys
from typing import (
    List,
    Any,
    Optional,
    Union,
    Tuple,
    Dict,
    ClassVar
)
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field

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

# Define model categories
class ModelType(Enum):
    CHAT = auto()
    RERANKER = auto()
    VISION = auto()
    EMBEDDING = auto()
    AUDIO = auto()
    TRANSCRIPTION = auto()

# Base model configuration
@dataclass
class ModelConfig:
    """Base configuration for any model"""
    name: str
    model_id: str  # The actual model ID used by the provider
    provider: str
    type: ModelType
    description: str = ""
    requires_api_key: bool = False
    api_key_name: Optional[str] = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.requires_api_key and not self.api_key_name:
            raise ValueError(f"Model {self.name} requires API key but no key name specified")

# Base class for all model providers
class ModelProvider(ABC):
    """Abstract base class for model providers"""
    model_configs: ClassVar[Dict[str, ModelConfig]] = {}
    
    @classmethod
    def register_model(cls, config: ModelConfig):
        """Register a model configuration"""
        cls.model_configs[config.name] = config
        logger.debug(f"Registered model: {config.name} from provider {config.provider}")
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name"""
        return cls.model_configs.get(model_name)
    
    @classmethod
    def list_models(cls, model_type: Optional[ModelType] = None) -> List[str]:
        """List all registered models, optionally filtered by type"""
        if model_type:
            return [name for name, config in cls.model_configs.items() 
                   if config.type == model_type]
        return list(cls.model_configs.keys())
    
    @abstractmethod
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create a model instance from configuration"""
        pass

# Main model factory
class ModelFactory:
    """Factory for creating language model instances"""
    _providers: Dict[str, ModelProvider] = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, provider: ModelProvider):
        """Register a model provider"""
        cls._providers[provider_name] = provider
        logger.info(f"Registered provider: {provider_name}")
    
    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> Any:
        """Create a language model instance by name"""
        # Find the provider that has this model registered
        for provider in cls._providers.values():
            config = provider.get_model_config(model_name)
            if config:
                # Get API key if needed
                if config.requires_api_key:
                    load_credentials()
                    api_key = get_api_key(config.api_key_name)
                    kwargs['api_key'] = api_key
                
                # Merge default parameters with provided ones
                merged_kwargs = {**config.default_params, **kwargs}
                
                # Create and return the model
                return provider.create_model(config, **merged_kwargs)
                
        # If we get here, no provider had this model
        raise ValueError(f"Unknown model: {model_name}. Available models: {cls.list_models()}")
    
    @classmethod
    def list_models(cls, model_type: Optional[ModelType] = None) -> List[str]:
        """List all available models, optionally filtered by type"""
        all_models = []
        for provider in cls._providers.values():
            all_models.extend(provider.list_models(model_type))
        return all_models
    
    @classmethod
    async def update_model(cls, new_model_name: str, current_model_name: str) -> Optional[Any]:
        """Get new model instance if choice has changed"""
        if new_model_name != current_model_name:
            return cls.get_model(new_model_name)
        return None

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
    
    # Test model support for testing environment
    if choice == "test_model" and os.getenv("TESTING") == "true":
        from langchain.schema import BaseMessage
        
        class TestModel:
            def __init__(self, response="Test response"):
                self.response = response
                
            def bind(self, **kwargs):
                return self
                
            async def astream(self, messages: List[BaseMessage], **kwargs):
                yield AIMessage(content=self.response)
                
            async def agenerate(self, messages: List[BaseMessage], **kwargs):
                return MockLLMResult(generations=[[MockGeneration(text=self.response)]])
                
        return TestModel()
    
    elif choice == "Ollama (llama3.2)":
        return ChatOllama(
            model="llama3.2:latest",
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
            convert_messages_to_human=True,
            **kwargs
        )
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
    elif choice == "Ollama (phi4)":
        return ChatOllama(
            model="phi4:latest",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (qwen2.5:14b)":
        return ChatOllama(
            model="qwen2.5:14b",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (llava)":
        return ChatOllama(
            model="llava:latest",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "Ollama (llava:7b-v1.6)":
        return ChatOllama(
            model="llava:7b-v1.6",
            base_url="http://localhost:11434",
            **kwargs)
    elif choice == "OpenAI o3-mini":
        api_key = get_api_key('openai')
        return ChatOpenAI(
            model="o3-mini", 
            api_key=api_key, 
            **kwargs)
    elif choice == "OpenAI GPT-4o-mini":
        api_key = get_api_key('openai')
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=api_key, 
            **kwargs)
    elif choice == "OpenAI GPT-4o":
        api_key = get_api_key('openai')
        return ChatOpenAI(
            model="gpt-4o", 
            api_key=api_key, 
            **kwargs)
    else:
        raise ValueError(f"Invalid model choice: {choice}")

async def update_model(new_choice: str, current_choice: str) -> Optional[Any]:
    """Get new model instance if choice has changed."""
    # Add type checking
    if not isinstance(new_choice, str):
        raise TypeError(f"Expected string for new_choice, got {type(new_choice).__name__}")
    if not isinstance(current_choice, str):
        raise TypeError(f"Expected string for current_choice, got {type(current_choice).__name__}")
        
    if new_choice != current_choice:
        return get_model(new_choice)
    return None

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

def get_reranker(model_name: str, **kwargs):
    """Get a reranker model by name"""
    model = ModelFactory.get_model(model_name, **kwargs)
    
    # Verify it's a reranker
    for provider in ModelFactory._providers.values():
        config = provider.get_model_config(model_name)
        if config and config.type == ModelType.RERANKER:
            return model
    
    raise ValueError(f"Model {model_name} is not a reranker model")

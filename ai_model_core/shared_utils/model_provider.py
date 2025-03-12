# ai_model_core/shared_utils/model_providers.py
from typing import Dict, Any, Optional
import logging

# Third-party imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatCohere
from langchain_cohere import ChatCohere, CohereEmbeddings
import cohere
import whisper
import mistralai

# Local imports
from .factory import ModelProvider, ModelConfig, ModelType

logger = logging.getLogger(__name__)

class OllamaProvider(ModelProvider):
    """Provider for Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create an Ollama model instance"""
        if config.type == ModelType.CHAT:
            return ChatOllama(
                model=config.model_id,
                base_url=self.base_url,
                **kwargs
            )
        elif config.type == ModelType.EMBEDDING:
            return OllamaEmbeddings(
                model=config.model_id,
                base_url=self.base_url,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for Ollama: {config.type}")

class CohereProvider(ModelProvider):
    """Provider for Cohere models"""
    
    def __init__(self):
        self.api_base = "https://api.cohere.ai/v1"
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create a Cohere model instance"""
        api_key = kwargs.pop('api_key', None)
        
        if config.type == ModelType.CHAT:
            # Handle chat models
            return ChatCohere(
                model=config.model_id,
                cohere_api_key=api_key,
                **kwargs
            )
        elif config.type == ModelType.EMBEDDING:
            # Handle embedding models
            return CohereEmbeddings(
                model=config.model_id,
                cohere_api_key=api_key,
                **kwargs
            )
        elif config.type == ModelType.RERANKER:
            # Handle reranker models
            return CohereRerank(
                model=config.model_id,
                cohere_api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for Cohere: {config.type}")
        
class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models"""
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create an OpenAI model instance"""
        api_key = kwargs.pop('api_key', None)
        
        # Handle OpenRouter base URL if specified
        base_url = kwargs.pop('base_url', None)
        model_kwargs = kwargs.pop('model_kwargs', {})
        
        # Add HTTP headers if provided
        if 'http_referer' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["HTTP-Referer"] = kwargs.pop('http_referer')
            
        if 'x_title' in kwargs:
            model_kwargs["headers"] = model_kwargs.get("headers", {})
            model_kwargs["headers"]["X-Title"] = kwargs.pop('x_title')
        
        if config.type == ModelType.CHAT:
            return ChatOpenAI(
                model=config.model_id,
                api_key=api_key,
                base_url=base_url,
                model_kwargs=model_kwargs,
                **kwargs
            )
        elif config.type == ModelType.EMBEDDING:
            return OpenAIEmbeddings(
                model=config.model_id,
                api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for OpenAI: {config.type}")
class AnthropicProvider(ModelProvider):
    """Provider for Anthropic models"""
    
    def __init__(self, provider_name: str = "Anthropic"):
        self.provider_name = provider_name
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create an Anthropic model instance"""
        api_key = kwargs.pop('api_key', None)
        
        if config.type == ModelType.CHAT:
            return ChatAnthropic(
                model=config.model_id,
                anthropic_api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for Anthropic: {config.type}")
class MistralProvider(ModelProvider):
    """Provider for Mistral AI models"""
    
    def __init__(self):
        self.base_url = "https://api.mistral.ai/v1"
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create a Mistral model instance"""
        api_key = kwargs.pop('api_key', None)
        
        if config.type == ModelType.CHAT:
            # Mistral uses OpenAI-compatible API
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_id,
                api_key=api_key,
                base_url=self.base_url,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for Mistral: {config.type}")

class GoogleProvider(ModelProvider):
    """Provider for Google models"""
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create a Google model instance"""
        api_key = kwargs.pop('api_key', None)
        
        if config.type == ModelType.CHAT:
            return ChatGoogleGenerativeAI(
                model=config.model_id,
                google_api_key=api_key,
                convert_messages_to_human=True,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for Google: {config.type}")

class WhisperProvider(ModelProvider):
    """Provider for Whisper models"""
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create a Whisper model instance"""
        if config.type == ModelType.TRANSCRIPTION:
            try:
                return whisper.load_model(config.model_id)
            except Exception as e:
                raise ValueError(f"Failed to load Whisper model: {str(e)}")
        else:
            raise ValueError(f"Unsupported model type for Whisper: {config.type}")

# Add custom provider for specific embedding models if needed
class CustomEmbeddingProvider(ModelProvider):
    """Provider for custom embedding models"""
    
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create a custom embedding model instance"""
        if config.type == ModelType.EMBEDDING and config.model_id.startswith("e5-"):
            # Import locally to avoid circular import
            from ..model_helpers.embeddings import E5Embeddings
            return E5Embeddings(
                model_name=config.model_id,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model for CustomEmbedding: {config.model_id}")

# Create instances of the provider classes
ollama_provider = OllamaProvider()
anthropic_provider = AnthropicProvider()
cohere_provider = CohereProvider()
openai_provider = OpenAIProvider()
mistral_provider = MistralProvider()
google_provider = GoogleProvider()
whisper_provider = WhisperProvider()
custom_embedding_provider = CustomEmbeddingProvider()